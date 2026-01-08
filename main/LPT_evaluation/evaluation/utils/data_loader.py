"""
Dataset Loader for Evaluation Framework

Handles loading and preprocessing of ProverQA and NSA-LR datasets.
Supports both JSON (ProverQA) and JSONL (FOLIO) formats.
Converts data to standardized evaluation format with complexity analysis.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import our complexity analyzer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from complexity_analyzer import ComplexityAnalyzer, FOLIOComplexity

logger = logging.getLogger(__name__)


@dataclass
class FOLIOExample:
    """Standardized example for evaluation"""
    example_id: str
    story_id: int
    premises: str
    premises_fol: str
    conclusion: str
    conclusion_fol: str
    label: str  # "True", "False", "Uncertain"
    complexity: Optional[FOLIOComplexity] = None
    reasoning: Optional[str] = None  # Store reasoning for hop count calculation


def count_hops(reasoning: str) -> int:
    """
    Count the number of reasoning hops (number of conclusions).

    Args:
        reasoning: Reasoning process text

    Returns:
        Number of hops (count of conclusion markers)
    """
    if not reasoning or not reasoning.strip():
        return 0

    # Count "conclusion:" pattern occurrences
    pattern = r'\bconclusion\s*:'
    matches = re.findall(pattern, reasoning.lower())

    return len(matches)
    

class FOLIODataLoader:
    """Data loader for ProverQA and NSA-LR datasets"""
    
    def __init__(self, data_path: str, include_complexity: bool = True):
        self.data_path = Path(data_path)
        self.include_complexity = include_complexity
        
        if include_complexity:
            self.complexity_analyzer = ComplexityAnalyzer()
            logger.info("Complexity analyzer initialized")
        
        # Validate data file
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Detect data format
        self.is_proverqa = self._detect_proverqa_format()
    
    def _detect_proverqa_format(self) -> bool:
        """Detect if the data file is in ProverQA format"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB
                if content.strip().startswith('['):
                    # Likely JSON array format (ProverQA)
                    return True
                elif '{' in content and 'example_id' in content:
                    # Likely JSONL format (FOLIO)
                    return False
                # Default to FOLIO if unsure
                return False
        except Exception:
            return False
    
    def load_data(self, max_examples: Optional[int] = None) -> List[FOLIOExample]:
        """Load data from JSONL file (FOLIO) or JSON file (ProverQA)"""
        logger.info(f"Loading data from {self.data_path} (format: {'ProverQA' if self.is_proverqa else 'FOLIO'})")
        
        if self.is_proverqa:
            return self._load_proverqa_data(max_examples)
        else:
            return self._load_folio_data(max_examples)
    
    def _load_folio_data(self, max_examples: Optional[int] = None) -> List[FOLIOExample]:
        """Load FOLIO data from JSONL file"""
        examples = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_examples and len(examples) >= max_examples:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        raw_example = json.loads(line)
                        example = self._parse_folio_example(raw_example)
                        examples.append(example)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to process example at line {line_num}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise
        
        logger.info(f"Loaded {len(examples)} FOLIO examples")
        
        # Add complexity analysis if requested
        if self.include_complexity and examples:
            logger.info("Computing complexity scores for examples...")
            self._add_complexity_scores(examples)
        
        return examples
    
    def _load_proverqa_data(self, max_examples: Optional[int] = None) -> List[FOLIOExample]:
        """Load ProverQA data from JSON file"""
        examples = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for i, raw_example in enumerate(raw_data):
                if max_examples and len(examples) >= max_examples:
                    break
                    
                try:
                    example = self._parse_proverqa_example(raw_example)
                    examples.append(example)
                    
                except Exception as e:
                    logger.warning(f"Failed to process ProverQA example {i}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load ProverQA data from {self.data_path}: {e}")
            raise
        
        logger.info(f"Loaded {len(examples)} ProverQA examples")
        
        # Add complexity analysis if requested
        if self.include_complexity and examples:
            logger.info("Computing complexity scores for examples...")
            self._add_complexity_scores(examples)
        
        return examples
    
    def _parse_folio_example(self, raw_example: Dict[str, Any]) -> FOLIOExample:
        """Parse raw FOLIO example into standardized format"""
        return FOLIOExample(
            example_id=str(raw_example.get('example_id', '')),
            story_id=int(raw_example.get('story_id', 0)),
            premises=raw_example.get('premises', ''),
            premises_fol=raw_example.get('premises-FOL', ''),
            conclusion=raw_example.get('conclusion', ''),
            conclusion_fol=raw_example.get('conclusion-FOL', ''),
            label=raw_example.get('label', 'Uncertain'),
            reasoning=raw_example.get('reasoning', '')  # FOLIO typically doesn't have reasoning, but add for consistency
        )
    
    def _parse_proverqa_example(self, raw_example: Dict[str, Any]) -> FOLIOExample:
        """Parse raw ProverQA example into standardized format"""
        # Map ProverQA answer format (A/B/C) to FOLIO labels
        answer_mapping = {
            'A': 'True',
            'B': 'False', 
            'C': 'Uncertain'
        }
        
        # Extract FOL formulas from nl2fol if available
        premises_fol = []
        conclusion_fol = raw_example.get('conclusion_fol', '')

        if 'nl2fol' in raw_example:
            nl2fol = raw_example['nl2fol']
            if isinstance(nl2fol, dict):
                # Extract all FOL translations as a list of premises
                for key, fol_expr in nl2fol.items():
                    if fol_expr and fol_expr.strip():
                        premises_fol.append(fol_expr)

        premises_unified = raw_example.get('context_unified')
        if isinstance(premises_unified, str) and premises_unified.strip():
            premises_value = premises_unified
        else:
            premises_value = raw_example.get('context', '')

        question_unified = raw_example.get('question')
        if isinstance(question_unified, str) and question_unified.strip():
            conclusion_value = question_unified
        else:
            conclusion_value = raw_example.get('question', '')

        return FOLIOExample(
            example_id=str(raw_example.get('id', '')),
            story_id=int(raw_example.get('id', 0)),  # Use id as story_id for ProverQA
            premises=premises_value,
            premises_fol=premises_fol,
            conclusion=conclusion_value,
            conclusion_fol=conclusion_fol,
            label=answer_mapping.get(raw_example.get('answer', 'C'), 'Uncertain'),
            reasoning=raw_example.get('reasoning', '')  # Store reasoning for hop count calculation
        )
    
    def _add_complexity_scores(self, examples: List[FOLIOExample]) -> None:
        """Add complexity analysis to examples"""
        for i, example in enumerate(examples):
            try:
                # Calculate hop count from reasoning if available
                hop_count = 0
                if example.reasoning:
                    hop_count = count_hops(example.reasoning)

                # Create a sample dict in the format expected by complexity analyzer
                sample = {
                    'example_id': example.example_id,
                    'premises-FOL': example.premises_fol,
                    'conclusion-FOL': example.conclusion_fol,
                    'label': example.label
                }

                # Skip examples without FOL data
                if not example.premises_fol and not example.conclusion_fol:
                    logger.warning(f"Skipping example {example.example_id}: No FOL data available")
                    continue

                # Analyze complexity with hop count
                complexity = self.complexity_analyzer.analyze_folio_sample(sample, hop_count=hop_count)
                example.complexity = complexity

                if (i + 1) % 50 == 0:
                    logger.info(f"Computed complexity for {i + 1}/{len(examples)} examples")

            except Exception as e:
                logger.warning(f"Failed to compute complexity for example {example.example_id}: {e}")
    
    def get_examples_by_label(self, examples: List[FOLIOExample], label: str) -> List[FOLIOExample]:
        """Filter examples by label"""
        return [ex for ex in examples if ex.label.lower() == label.lower()]
    
    def get_examples_by_complexity(self, examples: List[FOLIOExample], 
                                 min_complexity: float = 0, 
                                 max_complexity: float = float('inf')) -> List[FOLIOExample]:
        """Filter examples by total complexity score"""
        filtered = []
        for ex in examples:
            if ex.complexity and ex.complexity.total_complexity:
                total_score = ex.complexity.total_complexity.total
                if min_complexity <= total_score <= max_complexity:
                    filtered.append(ex)
        return filtered
    
    def get_complexity_statistics(self, examples: List[FOLIOExample]) -> Dict[str, Any]:
        """Get complexity statistics for the dataset"""
        if not any(ex.complexity for ex in examples):
            return {"error": "No complexity data available"}
        
        # Extract complexity scores
        total_scores = []
        semantic_scores = []
        structural_scores = []
        premises_scores = []
        conclusion_scores = []
        
        for ex in examples:
            if ex.complexity:
                total_scores.append(ex.complexity.total_complexity.total)
                semantic_scores.append(ex.complexity.total_complexity.semantic)
                structural_scores.append(ex.complexity.total_complexity.structural)
                premises_scores.append(ex.complexity.premises_complexity.total)
                conclusion_scores.append(ex.complexity.conclusion_complexity.total)
        
        def get_stats(scores):
            if not scores:
                return {}
            return {
                "min": min(scores),
                "max": max(scores),
                "mean": sum(scores) / len(scores),
                "count": len(scores)
            }
        
        return {
            "total_complexity": get_stats(total_scores),
            "semantic_complexity": get_stats(semantic_scores),
            "structural_complexity": get_stats(structural_scores),
            "premises_complexity": get_stats(premises_scores),
            "conclusion_complexity": get_stats(conclusion_scores),
            "label_distribution": self._get_label_distribution(examples)
        }
    
    def _get_label_distribution(self, examples: List[FOLIOExample]) -> Dict[str, int]:
        """Get distribution of labels in the dataset"""
        distribution = {}
        for ex in examples:
            label = ex.label
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def save_processed_data(self, examples: List[FOLIOExample], output_path: str) -> None:
        """Save processed examples to JSON file"""
        output_data = []
        
        for ex in examples:
            data = {
                "example_id": ex.example_id,
                "story_id": ex.story_id,
                "premises": ex.premises,
                "premises_fol": ex.premises_fol,
                "conclusion": ex.conclusion,
                "conclusion_fol": ex.conclusion_fol,
                "label": ex.label
            }
            
            # Add complexity data if available
            if ex.complexity:
                data["complexity"] = {
                    "premises": {
                        "semantic": ex.complexity.premises_complexity.semantic,
                        "structural": ex.complexity.premises_complexity.structural,
                        "total": ex.complexity.premises_complexity.total
                    },
                    "conclusion": {
                        "semantic": ex.complexity.conclusion_complexity.semantic,
                        "structural": ex.complexity.conclusion_complexity.structural,
                        "total": ex.complexity.conclusion_complexity.total
                    },
                    "overall": {
                        "semantic": ex.complexity.total_complexity.semantic,
                        "structural": ex.complexity.total_complexity.structural,
                        "total": ex.complexity.total_complexity.total
                    }
                }
            
            output_data.append(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(output_data)} processed examples to {output_path}")


class FOLIODataSplitter:
    """Utility for splitting FOLIO data by various criteria"""
    
    @staticmethod
    def split_by_complexity(examples: List[FOLIOExample], 
                          thresholds: List[float] = [20, 40, 60]) -> Dict[str, List[FOLIOExample]]:
        """Split examples into complexity buckets"""
        splits = {
            "low": [],
            "medium": [],
            "high": [],
            "very_high": []
        }
        
        for ex in examples:
            if not ex.complexity:
                continue
                
            total_complexity = ex.complexity.total_complexity.total
            
            if total_complexity < thresholds[0]:
                splits["low"].append(ex)
            elif total_complexity < thresholds[1]:
                splits["medium"].append(ex)
            elif total_complexity < thresholds[2]:
                splits["high"].append(ex)
            else:
                splits["very_high"].append(ex)
        
        return splits
    
    @staticmethod
    def split_by_label(examples: List[FOLIOExample]) -> Dict[str, List[FOLIOExample]]:
        """Split examples by label"""
        splits = {}
        for ex in examples:
            label = ex.label
            if label not in splits:
                splits[label] = []
            splits[label].append(ex)
        return splits
    
    @staticmethod
    def create_balanced_sample(examples: List[FOLIOExample], 
                             sample_size: int, 
                             balance_by: str = "label") -> List[FOLIOExample]:
        """Create a balanced sample of examples"""
        if balance_by == "label":
            splits = FOLIODataSplitter.split_by_label(examples)
            labels = list(splits.keys())
            per_label = sample_size // len(labels)
            
            balanced_sample = []
            for label in labels:
                label_examples = splits[label][:per_label]
                balanced_sample.extend(label_examples)
            
            return balanced_sample
        
        # Add other balancing strategies as needed
        return examples[:sample_size]


def create_evaluation_prompt(example: FOLIOExample, mode: str = "direct") -> Dict[str, str]:
    """Convert FOLIO example to evaluation prompt format"""
    if mode.lower() == "direct":
        prompt = f"""Context:
{example.premises}

Question: Based on the above information, is the following statement true, false, or uncertain? {example.conclusion}

Options:
A) True
B) False
C) Uncertain

The correct option is:"""
    
    elif mode.lower() == "cot":
        prompt = f"""Context:
{example.premises}

Question: Based on the above information, is the following statement true, false, or uncertain? {example.conclusion}

Options:
A) True
B) False
C) Uncertain

Please think step by step and provide your reasoning, then give your final answer.

The correct option is:"""
    
    else:
        raise ValueError(f"Unsupported prompt mode: {mode}")
    
    return {
        "prompt": prompt,
        "context": example.premises,
        "question": f"Based on the above information, is the following statement true, false, or uncertain? {example.conclusion}",
        "options": ["A) True", "B) False", "C) Uncertain"],
        "label": example.label,
        "example_id": example.example_id
    }


if __name__ == "__main__":
    # Test the data loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FOLIO data loader")
    parser.add_argument("--data_path", type=str, default="../data/folio_v2_validation.jsonl")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--include_complexity", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create loader and load data
    loader = FOLIODataLoader(args.data_path, args.include_complexity)
    examples = loader.load_data(args.max_examples)
    
    print(f"Loaded {len(examples)} examples")
    
    # Print first example
    if examples:
        ex = examples[0]
        print(f"\nExample {ex.example_id}:")
        print(f"Premises: {ex.premises[:200]}...")
        print(f"Conclusion: {ex.conclusion}")
        print(f"Label: {ex.label}")
        
        if ex.complexity:
            print(f"Complexity: {ex.complexity.total_complexity.total:.2f}")
        
        # Show evaluation prompt
        prompt_data = create_evaluation_prompt(ex, "direct")
        print(f"\nDirect prompt:\n{prompt_data['prompt']}")
    
    # Show statistics
    if args.include_complexity:
        stats = loader.get_complexity_statistics(examples)
        print(f"\nComplexity statistics: {stats}")
