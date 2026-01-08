"""
Logical Reasoning Evaluation Engine

Main evaluation framework that orchestrates:
- Data loading and preprocessing
- Model inference with multiple providers
- Complexity analysis integration
- Result collection and analysis

Supports ProverQA and NSA-LR datasets for evaluating logical reasoning capabilities.
"""

import os
import json
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Import our modules
from evaluation.models.llm_interface import LLMInterface, LLMResponse
from evaluation.utils.data_loader import FOLIODataLoader, FOLIOExample, FOLIODataSplitter
from evaluation.prompts.prompt_templates import PromptManager

# Import complexity analyzer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from complexity_analyzer import ComplexityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional json_repair integration
try:
    from json_repair import repair_json  # type: ignore
    _JSON_REPAIR_AVAILABLE = True
except Exception:
    _JSON_REPAIR_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    example_id: str
    story_id: int
    context: str
    question: str
    ground_truth: str
    model_response: str
    predicted_answer: Optional[str]
    is_correct: Optional[bool]
    reasoning: Optional[str] = None
    complexity_score: Optional[float] = None
    response_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""
    total_examples: int
    correct_predictions: int
    accuracy: float
    accuracy_by_label: Dict[str, float]
    accuracy_by_complexity: Dict[str, float]
    avg_response_time: float
    error_count: int
    timestamp: str


class FOLIOEvaluator:
    """Main logical reasoning evaluation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._initialize_model()
        self.prompt_manager = PromptManager()
        self.data_loader = FOLIODataLoader(
            config['data_path'], 
            include_complexity=config.get('include_complexity', True)
        )
        # Debug flag (can be enabled via config['debug']=True or log level DEBUG)
        self.debug = bool(config.get('debug')) or logger.isEnabledFor(logging.DEBUG)
        
        # Setup output directory
        self.output_dir = Path(config.get('output_dir', 'results/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluator initialized with {config['provider']} - {config['model_name']}")
    
    def _initialize_model(self) -> LLMInterface:
        """Initialize LLM interface based on config"""
        provider = self.config['provider']
        model_name = self.config['model_name']
        provider_config = self.config.get('provider_config', {})
        
        return LLMInterface(provider, model_name, **provider_config)
    
    def evaluate(self, 
                max_examples: Optional[int] = None,
                start_idx: int = 0,
                end_idx: Optional[int] = None,
                filter_by_complexity: Optional[Tuple[float, float]] = None,
                filter_by_label: Optional[str] = None) -> Tuple[List[EvaluationResult], EvaluationSummary]:
        """Run evaluation on dataset"""

        # Load data
        logger.info("Loading dataset...")
        examples = self.data_loader.load_data(max_examples)
        
        # Apply filters
        if filter_by_complexity:
            min_complexity, max_complexity = filter_by_complexity
            examples = self.data_loader.get_examples_by_complexity(
                examples, min_complexity, max_complexity
            )
            logger.info(f"Filtered to {len(examples)} examples by complexity ({min_complexity}-{max_complexity})")
        
        if filter_by_label:
            examples = self.data_loader.get_examples_by_label(examples, filter_by_label)
            logger.info(f"Filtered to {len(examples)} examples with label '{filter_by_label}'")
        
        # Apply index range
        if end_idx is None:
            end_idx = len(examples)
        examples = examples[start_idx:end_idx]
        
        logger.info(f"Evaluating {len(examples)} examples (indices {start_idx}-{end_idx})")
        
        # Run evaluation
        results = []
        total_time = 0
        
        with tqdm(total=len(examples), desc="Evaluating") as pbar:
            for example in examples:
                try:
                    start_time = time.time()
                    result = self._evaluate_single_example(example)
                    end_time = time.time()
                    
                    result.response_time = end_time - start_time
                    total_time += result.response_time
                    
                    results.append(result)
                    
                    # Update progress
                    pbar.set_postfix({
                        'accuracy': f"{self._calculate_accuracy(results):.3f}",
                        'avg_time': f"{total_time/len(results):.2f}s"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate example {example.example_id}: {e}")
                    # Create error result
                    error_result = EvaluationResult(
                        example_id=example.example_id,
                        story_id=example.story_id,
                        context=example.premises,
                        question=f"Based on the above information, is the following statement true, false, or uncertain? {example.conclusion}",
                        ground_truth=example.label,
                        model_response="",
                        predicted_answer=None,
                        is_correct=False,
                        complexity_score=example.complexity.total_complexity.total if example.complexity else None,
                        error=str(e)
                    )
                    results.append(error_result)
                    pbar.update(1)
        
        # Generate summary
        summary = self._generate_summary(results, examples)
        
        # Save results
        self._save_results(results, summary)
        
        return results, summary
    
    def _evaluate_single_example(self, example: FOLIOExample) -> EvaluationResult:
        """Evaluate a single example"""
        
        # Prepare prompt
        template_name = self.config.get('prompt_template', 'json_direct')
        context = example.premises

        raw_question = example.conclusion or ""
        if re.search(r"based on the above information", raw_question, re.IGNORECASE):
            question = raw_question
        else:
            question = f"Based on the above information, is the following statement true, false, or uncertain? {raw_question}".strip()
        options = ["A) True", "B) False", "C) Uncertain"]
        
        # Format prompt for the model
        if self.config.get('use_chat_format', True):
            messages = self.prompt_manager.format_for_chat_model(template_name, context, question, options)
            prompt = messages
        else:
            template = self.prompt_manager.get_template(template_name)
            prompt = template.create_full_prompt(context, question, options)
        
        # Get model response
        response = self.model.completion(
            prompt,
            temperature=self.config.get('temperature', 0.0),
            max_tokens=self.config.get('max_tokens', 512)
        )

        if self.debug:
            logger.debug("=" * 80)
            logger.debug(f"[EVAL][RAW RESPONSE] example_id={example.example_id} template={template_name} provider={response.provider} model={response.model_name}")
            logger.debug(f"Prompt Type={'chat' if isinstance(prompt, list) else 'text'}")
            if isinstance(prompt, list):
                try:
                    logger.debug(f"Prompt Messages Count={len(prompt)}")
                except Exception:
                    pass
            raw_txt = response.response_text or ""
            logger.debug(f"Raw Response Length={len(raw_txt)}")
            logger.debug(f"Raw Response Preview (first 400 chars)\n{raw_txt[:400]}")
            if len(raw_txt) > 400:
                logger.debug(f"Raw Response Tail (last 400 chars)\n{raw_txt[-400:]}")
        
        # Parse response
        predicted_answer, reasoning, parse_error = self._parse_response(response.response_text, template_name)

        # Compose a complete model_response for saving: ensure think is closed and final answer visible
        saved_response_text = response.response_text or ""
        try:
            if saved_response_text:
                # If there's an opening <think> without a closing tag, close it to keep structure intact
                if '<think>' in saved_response_text and '</think>' not in saved_response_text:
                    saved_response_text = saved_response_text.rstrip() + "\n</think>\n"
                # If we successfully parsed an answer but the tail lacks an explicit final answer marker, append one
                if predicted_answer:
                    tail = saved_response_text[-600:].lower()
                    has_final_marker = ('final answer' in tail) or ('"answer"' in tail) or ('答案' in tail) or ('结论' in tail)
                    if not has_final_marker:
                        saved_response_text = saved_response_text.rstrip() + f"\n\nFinal answer: {predicted_answer}"
        except Exception:
            # In case of any unexpected errors, fall back to original response text
            saved_response_text = response.response_text or ""

        if self.debug:
            logger.debug(f"[EVAL][PARSE RESULT] example_id={example.example_id} predicted={predicted_answer} reasoning_len={len(reasoning) if reasoning else 0} parse_error={parse_error}")
            if saved_response_text != (response.response_text or ""):
                logger.debug("[EVAL][MODEL_RESPONSE ADJUSTED] Added closing </think> or Final answer tag.")
            logger.debug(f"[EVAL][MODEL_RESPONSE LENGTH] {len(saved_response_text)}")
            logger.debug(f"[EVAL][MODEL_RESPONSE PREVIEW] {saved_response_text[:300]}")
            logger.debug("=" * 80)
        
        # Check correctness
        is_correct = self._is_correct(predicted_answer, example.label) if predicted_answer else False
        
        # Get complexity score
        complexity_score = None
        if example.complexity:
            complexity_score = example.complexity.total_complexity.total
        
        return EvaluationResult(
            example_id=example.example_id,
            story_id=example.story_id,
            context=context,
            question=question,
            ground_truth=example.label,
            model_response=saved_response_text,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            reasoning=reasoning,
            complexity_score=complexity_score,
            error=parse_error
        )
    
    def _parse_response(self, response_text: str, template_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse model response to extract answer and reasoning"""
        try:
            raw = response_text or ""
            text = raw

            # 1) Remove code fences to simplify parsing
            text = re.sub(r"```[a-zA-Z]*\n", "", text)
            text = text.replace("```", "")

            # 2) Remove <think>...</think> or similar analysis blocks
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text, flags=re.IGNORECASE)

            cleaned_text = text.strip()

            # Helper: extract last valid JSON object that contains an answer-like field
            def find_json_candidates(s: str) -> List[str]:
                candidates = []
                stack = []
                start_idx = None
                for i, ch in enumerate(s):
                    if ch == '{':
                        if not stack:
                            start_idx = i
                        stack.append('{')
                    elif ch == '}':
                        if stack:
                            stack.pop()
                            if not stack and start_idx is not None:
                                candidates.append(s[start_idx:i+1])
                                start_idx = None
                return candidates

            # Helper to attempt parse & extraction (optionally using json_repair)
            def attempt_parse(js_fragment: str) -> Tuple[Optional[str], Optional[str], bool]:
                """Return (answer, reasoning, success)."""
                parsed_obj = None
                # First try normal json
                try:
                    parsed_obj = json.loads(js_fragment)
                except Exception:
                    # If template suggests JSON and json_repair available, try repair
                    if _JSON_REPAIR_AVAILABLE and template_name.startswith('json'):
                        try:
                            repaired = repair_json(js_fragment, ensure_ascii=False)
                            parsed_obj = json.loads(repaired)
                        except Exception:
                            return None, None, False
                    else:
                        return None, None, False

                if not isinstance(parsed_obj, dict):
                    return None, None, False

                answer_raw: Optional[str] = None
                for key in ["answer", "final_answer", "prediction", "choice", "label"]:
                    if key in parsed_obj:
                        answer_raw = str(parsed_obj.get(key))
                        break
                if not answer_raw:
                    # nested containers
                    for container in ["data", "result", "output"]:
                        sub = parsed_obj.get(container)
                        if isinstance(sub, dict):
                            for key in ["answer", "final_answer", "prediction", "choice", "label"]:
                                if key in sub:
                                    answer_raw = str(sub.get(key))
                                    break
                        if answer_raw:
                            break
                if not answer_raw:
                    return None, None, False

                reasoning_val = None
                for rkey in ["reasoning", "rationale", "explanation"]:
                    if rkey in parsed_obj:
                        reasoning_val = str(parsed_obj.get(rkey))
                        break

                normalized = self._normalize_answer(answer_raw)
                if not normalized:
                    return None, None, False
                return normalized, reasoning_val, True

            # 3) Try JSON candidates (from last to first)
            json_candidates = find_json_candidates(cleaned_text)
            for idx, js in enumerate(reversed(json_candidates)):
                ans, reas, ok = attempt_parse(js)
                if self.debug:
                    logger.debug(f"[PARSE][JSON CANDIDATE] idx={idx} len={len(js)} success={ok} answer={ans} reasoning_len={len(reas) if reas else 0}")
                if ok:
                    return ans, reas, None

            # 3b) If template is json_* and we failed above, try repairing the entire trailing portion
            if template_name.startswith('json') and _JSON_REPAIR_AVAILABLE:
                # Heuristic: grab substring from last '{' to end
                last_brace = cleaned_text.rfind('{')
                if last_brace != -1:
                    fragment = cleaned_text[last_brace:]
                    ans, reas, ok = attempt_parse(fragment)
                    if self.debug:
                        logger.debug(f"[PARSE][WHOLE TRAILING REPAIR] success={ok} answer={ans}")
                    if ok:
                        return ans, reas, None

            # 4) Try to capture explicit "final answer" lines (search from end)
            tail_window = cleaned_text[-600:]  # focus on the end
            patterns = [
                r"(?i)(?:final\s*answer|答案|最终答案|结论|answer)\s*[:：]\s*([ABC](?:\)|\.|\s|$)|True|False|Uncertain|真|假|不确定)",
                r"(?i)(?:选择|选项|选)\s*[:：]?\s*([ABC](?:\)|\.|\s|$))",
                r"(?i)(?:therefore|thus|so)[^\n]*\b(is|be|=)?\s*(True|False|Uncertain|真|假|不确定)\b",
            ]
            last_match_val = None
            for pat in patterns:
                matches = list(re.finditer(pat, tail_window))
                if matches:
                    m = matches[-1]
                    # pick last non-empty capturing group from right
                    groups = [g for g in m.groups() if g]
                    if groups:
                        last_match_val = groups[-1]
            if last_match_val:
                normalized = self._normalize_answer(last_match_val)
                if self.debug:
                    logger.debug(f"[PARSE][FINAL ANSWER REGEX] raw_match={last_match_val} normalized={normalized}")
                if normalized:
                    return normalized, None, None

            # 5) Prefer the last occurrence of option tokens A)/B)/C) near the end
            option_tokens = ["A)", "B)", "C)", "(A)", "(B)", "(C)","A", "B", "C"]
            found = None
            for token in option_tokens:
                idx = tail_window.rfind(token)
                if idx != -1:
                    # keep the one appearing latest
                    if not found or idx > found[1]:
                        found = (token, idx)
            if found:
                norm = self._normalize_answer(found[0])
                if self.debug:
                    logger.debug(f"[PARSE][OPTION TOKEN FALLBACK] token={found[0]} normalized={norm}")
                return norm, None, None

            # 6) As a final fallback, pick the last occurrence of True/False/Uncertain (word boundaries)
            tfu_pattern = re.compile(r"\b(True|False|Uncertain|真|假|不确定)\b", re.IGNORECASE)
            matches = list(tfu_pattern.finditer(tail_window))
            if matches:
                normalized = self._normalize_answer(matches[-1].group(1))
                if self.debug:
                    logger.debug(f"[PARSE][TFU FALLBACK] matched={matches[-1].group(1)} normalized={normalized}")
                if normalized:
                    return normalized, None, None

            return None, None, f"Could not parse answer from: {cleaned_text[:200]}..."

        except Exception as e:
            return None, None, f"Parse error: {str(e)}"
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to standard format"""
        if not answer:
            return ""
        a = str(answer).strip()

        # Remove trailing punctuation
        a = re.sub(r"[\s\.]$", "", a)

        # Normalize some common wrappers
        a_upper = a.upper()

        # Map option letters to labels
        if a_upper in ['A', 'A)', '(A)', 'OPTION A', '选A', '选择A', 'A']:
            return 'True'
        if a_upper in ['B', 'B)', '(B)', 'OPTION B', '选B', '选择B', 'B']:
            return 'False'
        if a_upper in ['C', 'C)', '(C)', 'OPTION C', '选C', '选择C', 'C']:
            return 'Uncertain'

        # Map textual labels
        if a_upper in ['TRUE', 'T', '真', '正确']:
            return 'True'
        if a_upper in ['FALSE', 'F', '假', '错误']:
            return 'False'
        if a_upper in ['UNCERTAIN', 'UNKNOWN', 'U', '不确定']:
            return 'Uncertain'

        # Handle composite like "A) True" or "B) False"
        m = re.search(r"(?i)\b([ABC])\b.*\b(True|False|Uncertain)\b", a)
        if m:
            letter = m.group(1).upper()
            return {'A': 'True', 'B': 'False', 'C': 'Uncertain'}[letter]

        # If it directly says the label
        m2 = re.search(r"(?i)\b(True|False|Uncertain|真|假|不确定)\b", a)
        if m2:
            return self._normalize_answer(m2.group(1))

        return a
    
    def _is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth"""
        if not predicted:
            return False
        return predicted.lower() == ground_truth.lower()
    
    def _calculate_accuracy(self, results: List[EvaluationResult]) -> float:
        """Calculate accuracy from results"""
        if not results:
            return 0.0
        
        correct = sum(1 for r in results if r.is_correct)
        return correct / len(results)
    
    def _generate_summary(self, results: List[EvaluationResult], examples: List[FOLIOExample]) -> EvaluationSummary:
        """Generate evaluation summary"""
        
        # Overall accuracy
        total_examples = len(results)
        correct_predictions = sum(1 for r in results if r.is_correct)
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
        
        # Accuracy by label
        accuracy_by_label = {}
        for label in ['True', 'False', 'Uncertain']:
            label_results = [r for r in results if r.ground_truth == label]
            if label_results:
                label_correct = sum(1 for r in label_results if r.is_correct)
                accuracy_by_label[label] = label_correct / len(label_results)
            else:
                accuracy_by_label[label] = 0.0
        
        # Accuracy by complexity (if available)
        accuracy_by_complexity = {}
        if any(r.complexity_score for r in results):
            # Define complexity buckets
            complexity_buckets = [
                ("Low (0-20)", 0, 20),
                ("Medium (20-40)", 20, 40), 
                ("High (40-60)", 40, 60),
                ("Very High (60+)", 60, float('inf'))
            ]
            
            for bucket_name, min_val, max_val in complexity_buckets:
                bucket_results = [r for r in results 
                                if r.complexity_score and min_val <= r.complexity_score < max_val]
                if bucket_results:
                    bucket_correct = sum(1 for r in bucket_results if r.is_correct)
                    accuracy_by_complexity[bucket_name] = bucket_correct / len(bucket_results)
                else:
                    accuracy_by_complexity[bucket_name] = 0.0
        
        # Response time statistics
        response_times = [r.response_time for r in results if r.response_time]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Error count
        error_count = sum(1 for r in results if r.error)
        
        return EvaluationSummary(
            total_examples=total_examples,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            accuracy_by_label=accuracy_by_label,
            accuracy_by_complexity=accuracy_by_complexity,
            avg_response_time=avg_response_time,
            error_count=error_count,
            timestamp=datetime.now().isoformat()
        )
    
    def _save_results(self, results: List[EvaluationResult], summary: EvaluationSummary):
        """Save evaluation results and summary"""
        
        # Create filename with timestamp and model info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config['model_name'].replace('/', '_').replace(':', '_')
        template_name = self.config.get('prompt_template', 'default')
        
        # Create a dedicated folder for this evaluation run
        run_folder_name = f"eval_{self.config['provider']}_{model_name}_{template_name}_{timestamp}"
        run_folder = self.output_dir / run_folder_name
        run_folder.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = run_folder / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            results_data = [asdict(r) for r in results]
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = run_folder / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_file = run_folder / "report.txt"
        self._save_human_readable_report(results, summary, report_file)
        
        # Save configuration used for this run
        config_file = run_folder / "config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Results saved to {run_folder}")
        logger.info(f"  - Detailed results: {results_file.name}")
        logger.info(f"  - Summary: {summary_file.name}")
        logger.info(f"  - Report: {report_file.name}")
        logger.info(f"  - Configuration: {config_file.name}")
    
    def _save_human_readable_report(self, results: List[EvaluationResult], summary: EvaluationSummary, file_path: Path):
        """Save human-readable evaluation report"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Logical Reasoning Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            f.write(f"Model: {self.config['provider']} - {self.config['model_name']}\n")
            f.write(f"Timestamp: {summary.timestamp}\n")
            f.write(f"Template: {self.config.get('prompt_template', 'default')}\n\n")
            
            # Overall results
            f.write("Overall Results:\n")
            f.write(f"  Total Examples: {summary.total_examples}\n")
            f.write(f"  Correct Predictions: {summary.correct_predictions}\n")
            f.write(f"  Accuracy: {summary.accuracy:.3f}\n")
            f.write(f"  Average Response Time: {summary.avg_response_time:.2f}s\n")
            f.write(f"  Errors: {summary.error_count}\n\n")
            
            # Accuracy by label
            f.write("Accuracy by Label:\n")
            for label, acc in summary.accuracy_by_label.items():
                f.write(f"  {label}: {acc:.3f}\n")
            f.write("\n")
            
            # Accuracy by complexity
            if summary.accuracy_by_complexity:
                f.write("Accuracy by Complexity:\n")
                for complexity, acc in summary.accuracy_by_complexity.items():
                    f.write(f"  {complexity}: {acc:.3f}\n")
                f.write("\n")
            
            # Examples with errors
            error_results = [r for r in results if r.error]
            if error_results:
                f.write("Examples with Errors:\n")
                for r in error_results[:5]:  # Show first 5 errors
                    f.write(f"  {r.example_id}: {r.error}\n")
                if len(error_results) > 5:
                    f.write(f"  ... and {len(error_results) - 5} more\n")
                f.write("\n")
            
            # Sample predictions
            f.write("Sample Predictions:\n")
            correct_samples = [r for r in results if r.is_correct][:3]
            incorrect_samples = [r for r in results if not r.is_correct][:3]
            
            f.write("  Correct Predictions:\n")
            for i, r in enumerate(correct_samples, 1):
                f.write(f"    {i}. Example {r.example_id}: {r.ground_truth} -> {r.predicted_answer}\n")
            
            f.write("  Incorrect Predictions:\n")
            for i, r in enumerate(incorrect_samples, 1):
                f.write(f"    {i}. Example {r.example_id}: {r.ground_truth} -> {r.predicted_answer}\n")


if __name__ == "__main__":
    # Example usage
    config = {
        'provider': 'ollama',
        'model_name': 'qwen2.5:32b',
        'provider_config': {
            'base_url': 'http://localhost:11434',
            'temperature': 0.0,
            'max_tokens': 512
        },
        'data_path': '../data/folio_v2_validation.jsonl',
        'prompt_template': 'json_direct',
        'use_chat_format': True,
        'include_complexity': True,
        'output_dir': 'results/evaluation'
    }
    
    evaluator = FOLIOEvaluator(config)
    results, summary = evaluator.evaluate(max_examples=5)  # Test with 5 examples
    
    print(f"Evaluation completed!")
    print(f"Accuracy: {summary.accuracy:.3f}")
    print(f"Total examples: {summary.total_examples}")
    print(f"Errors: {summary.error_count}")
