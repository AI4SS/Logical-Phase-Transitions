"""
Prompt Templates for Logical Reasoning Evaluation

Supports Direct and Chain-of-Thought (CoT) prompting modes.
Compatible with ProverQA and NSA-LR datasets.
"""

import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PromptExample:
    """Single in-context example for few-shot prompting"""
    context: str
    question: str
    options: List[str]
    reasoning: Optional[str] = None  # For CoT examples
    answer: str = ""


class BasePromptTemplate(ABC):
    """Base class for prompt templates"""
    
    def __init__(self, examples: List[PromptExample] = None):
        self.examples = examples or []
    
    @abstractmethod
    def format_prompt(self, context: str, question: str, options: List[str]) -> str:
        """Format the main evaluation prompt"""
        pass
    
    @abstractmethod
    def format_example(self, example: PromptExample) -> str:
        """Format a single in-context example"""
        pass
    
    def create_full_prompt(self, context: str, question: str, options: List[str]) -> str:
        """Create complete prompt with examples + test case"""
        # Format in-context examples
        example_parts = []
        for example in self.examples:
            example_parts.append(self.format_example(example))
        
        # Add the test case
        test_prompt = self.format_prompt(context, question, options)
        
        if example_parts:
            return '\n------\n'.join(example_parts) + '\n------\n' + test_prompt
        else:
            return test_prompt


class DirectPromptTemplate(BasePromptTemplate):
    """Direct answer prompt template (no reasoning required)"""
    
    def format_prompt(self, context: str, question: str, options: List[str]) -> str:
        options_str = '\n'.join(options)
        return f"""Context:
{context}

Question: {question}

Options:
{options_str}

The correct option is:"""
    
    def format_example(self, example: PromptExample) -> str:
        options_str = '\n'.join(example.options)
        return f"""Context:
{example.context}

Question: {example.question}

Options:
{options_str}

The correct option is: {{
  "answer": "{example.answer}"
}}"""


class CoTPromptTemplate(BasePromptTemplate):
    """Chain-of-Thought prompt template (requires step-by-step reasoning)"""
    
    def format_prompt(self, context: str, question: str, options: List[str]) -> str:
        options_str = '\n'.join(options)
        return f"""Context:
{context}

Question: {question}

Options:
{options_str}

You are solving a logical problem.Please think step by step and provide your reasoning, then give your final answer.Please don't ouput extra symbols like \.

The correct option is:"""
    
    def format_example(self, example: PromptExample) -> str:
        options_str = '\n'.join(example.options)
        reasoning = example.reasoning or "Based on the given information, the answer is clear."
        
        return f"""Context:
{example.context}

Question: {example.question}

Options:
{options_str}

The correct option is: {{
  "reasoning": "{reasoning}",
  "answer": "{example.answer}"
}}"""


class JSONPromptTemplate(BasePromptTemplate):
    """JSON-structured prompt template for better parsing"""
    
    def __init__(self, mode: str = "direct", examples: List[PromptExample] = None):
        super().__init__(examples)
        self.mode = mode.lower()
    
    def format_prompt(self, context: str, question: str, options: List[str]) -> str:
        options_str = '\n'.join(options)
        
        if self.mode == "cot":
            instruction = "You are going to solve a logical problem.Please analyze the context step by step, then provide your reasoning and final answer in JSON format.Never forget to ouput the final answer in the end!"
            json_format = '{"reasoning": "your step-by-step analysis", "answer": "A|B|C"}'
        else:
            instruction = "You are going to solve a logical problem.Please provide your answer in JSON format."
            json_format = '{"answer": "A|B|C"}'

        return f"""Context:
{context}

Question: {question}

Options:
{options_str}

{instruction}

Format: {json_format}

Don't ouput extra symbols like \.
Response:"""
    
    def format_example(self, example: PromptExample) -> str:
        options_str = '\n'.join(example.options)
        
        if self.mode == "cot" and example.reasoning:
            response = f'{{"reasoning": "{example.reasoning}", "answer": "{example.answer}"}}'
        else:
            response = f'{{"answer": "{example.answer}"}}'
        
        return f"""Context:
{example.context}

Question: {example.question}

Options:
{options_str}

Response: {response}"""


class PromptTemplateFactory:
    """Factory for creating different prompt templates"""
    
    @staticmethod
    def create_template(template_type: str, mode: str = "direct", examples: List[PromptExample] = None) -> BasePromptTemplate:
        """Create prompt template of specified type"""
        template_type = template_type.lower()
        mode = mode.lower()
        
        if template_type == "direct":
            return DirectPromptTemplate(examples)
        elif template_type == "cot":
            return CoTPromptTemplate(examples)
        elif template_type == "json":
            return JSONPromptTemplate(mode, examples)
        else:
            raise ValueError(f"Unsupported template type: {template_type}")


class FOLIOPromptExamples:
    """Pre-defined examples for logical reasoning evaluation"""
    
    @staticmethod
    def get_direct_examples() -> List[PromptExample]:
        """Get examples for direct prompting"""
        return [
            PromptExample(
                context="is_progressive(Kaizen):::Kaizen's philosophy is progressive.\nembraces_fairness(Kaizen):::Kaizen's philosophy embraces fairness.\nis_progressive(Kaizen) → (embraces_fairness(Kaizen) ⊕ encourages_change(Kaizen)):::If Kaizen's philosophy is progressive, then it either embraces fairness or encourages change, but not both.\nadvocates_equity(Kaizen):::Kaizen advocates equity.\n(encourages_change(Kaizen) ⊕ advocates_equity(Kaizen)) → promotes_continuous_improvement(Kaizen):::If Kaizen's philosophy either encourages change or advocates equity (but not both), then it promotes continuous improvement.",
                question="Based on the above information, is the following statement true, false, or uncertain? Kaizen supports fair redistribution:::supports_fair_redistribution(Kaizen)",
                options=["A) True", "B) False", "C) Uncertain"],
                answer="C"
            ),
            PromptExample(
                context="expresses_style(Johanna):::Johanna expresses style.\n∀x (expresses_style(x) → (values_individuality(x) ∧ appreciates_art(x))):::If a person expresses style, then they value individuality and appreciate art.\nlikes_body_art(Johanna):::Johanna likes body art.\n(likes_body_art(Johanna) ∧ values_individuality(Johanna)) → has_septum_piercing(Johanna):::If Johanna likes body art and values individuality, then she has a septum piercing.\n∀x (has_septum_piercing(x) ⊕ does_not_have_septum_piercing(x)):::Every human either has a septum piercing or does not have a septum piercing, but not both.",
                question="Based on the above information, is the following statement true, false, or uncertain? Johanna does not not have a septum piercing:::¬does_not_have_septum_piercing(Johanna)",
                options=["A) True", "B) False", "C) Uncertain"],
                answer="A"
            )
        ]
    
    @staticmethod
    def get_cot_examples() -> List[PromptExample]:
        """Get examples for Chain-of-Thought prompting"""
        return [
            PromptExample(
                context="is_progressive(Kaizen):::Kaizen's philosophy is progressive.\nembraces_fairness(Kaizen):::Kaizen's philosophy embraces fairness.\nis_progressive(Kaizen) → (embraces_fairness(Kaizen) ⊕ encourages_change(Kaizen)):::If Kaizen's philosophy is progressive, then it either embraces fairness or encourages change, but not both.\nadvocates_equity(Kaizen):::Kaizen advocates equity.\n(encourages_change(Kaizen) ⊕ advocates_equity(Kaizen)) → promotes_continuous_improvement(Kaizen):::If Kaizen's philosophy either encourages change or advocates equity (but not both), then it promotes continuous improvement.",
                question="Based on the above information, is the following statement true, false, or uncertain? Kaizen supports fair redistribution:::supports_fair_redistribution(Kaizen)",
                options=["A) True", "B) False", "C) Uncertain"],
                reasoning="fact1: Kaizen's philosophy is progressive.\nfact2: Kaizen's philosophy embraces fairness.\nrule: If Kaizen's philosophy is progressive, then it either embraces fairness or encourages change, but not both.\nconclusion: Kaizen does not encourage change.\n\nfact1: Kaizen does not encourage change.\nfact2: Kaizen advocates equity.\nrule: If Kaizen's philosophy either encourages change or advocates equity (but not both), then it promotes continuous improvement.\nconclusion: Kaizen's philosophy promotes continuous improvement.\n\nAccording to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that Kaizen supports fair redistribution. The correct option is: C.",
                answer="C"
            ),
            PromptExample(
                context="expresses_style(Johanna):::Johanna expresses style.\n∀x (expresses_style(x) → (values_individuality(x) ∧ appreciates_art(x))):::If a person expresses style, then they value individuality and appreciate art.\nlikes_body_art(Johanna):::Johanna likes body art.\n(likes_body_art(Johanna) ∧ values_individuality(Johanna)) → has_septum_piercing(Johanna):::If Johanna likes body art and values individuality, then she has a septum piercing.\n∀x (has_septum_piercing(x) ⊕ does_not_have_septum_piercing(x)):::Every human either has a septum piercing or does not have a septum piercing, but not both.",
                question="Based on the above information, is the following statement true, false, or uncertain? Johanna does not not have a septum piercing:::¬does_not_have_septum_piercing(Johanna)",
                options=["A) True", "B) False", "C) Uncertain"],
                reasoning="fact1: Johanna expresses style.\nrule: If a person expresses style, then they value individuality and appreciate art.\nconclusion: Johanna values individuality.\n\nfact1: Johanna likes body art.\nfact2: Johanna values individuality.\nrule: If Johanna likes body art and values individuality, then she has a septum piercing.\nconclusion: Johanna has a septum piercing.\n\nfact1: Johanna has a septum piercing.\nrule: Every human either has a septum piercing or does not have a septum piercing, but not both.\nconclusion: Johanna does not not have a septum piercing.\n\nTherefore, it is true that Johanna does not not have a septum piercing. The correct option is: A.",
                answer="A"
            )
        ]
        


class PromptManager:
    """Manages different prompt templates and configurations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default template configurations"""
        # Direct templates
        self.templates["direct"] = PromptTemplateFactory.create_template(
            "direct", 
            examples=FOLIOPromptExamples.get_direct_examples()
        )
        
        # CoT templates
        self.templates["cot"] = PromptTemplateFactory.create_template(
            "cot",
            examples=FOLIOPromptExamples.get_cot_examples()
        )
        
        # JSON templates
        self.templates["json_direct"] = PromptTemplateFactory.create_template(
            "json", 
            mode="direct",
            examples=FOLIOPromptExamples.get_direct_examples()
        )
        
        self.templates["json_cot"] = PromptTemplateFactory.create_template(
            "json",
            mode="cot", 
            examples=FOLIOPromptExamples.get_cot_examples()
        )
    
    def get_template(self, template_name: str) -> BasePromptTemplate:
        """Get template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[template_name]
    
    def add_custom_template(self, name: str, template: BasePromptTemplate):
        """Add custom template"""
        self.templates[name] = template
    
    def create_system_prompt(self, mode: str = "direct") -> str:
        """Create system prompt for chat models"""
        if mode.lower() == "cot":
            return "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."
        else:
            return "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with key: answer."
    
    def format_for_chat_model(self, template_name: str, context: str, question: str, options: List[str]) -> List[Dict[str, str]]:
        """Format prompt for chat-based models (OpenAI, Claude, etc.)"""
        template = self.get_template(template_name)
        user_prompt = template.create_full_prompt(context, question, options)
        
        # Determine mode from template name
        mode = "cot" if "cot" in template_name.lower() else "direct"
        system_prompt = self.create_system_prompt(mode)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


def load_custom_examples_from_file(file_path: str) -> List[PromptExample]:
    """Load custom examples from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append(PromptExample(
            context=item['context'],
            question=item['question'],
            options=item['options'],
            reasoning=item.get('reasoning'),
            answer=item['answer']
        ))
    
    return examples


if __name__ == "__main__":
    # Test the prompt templates
    print("Testing Logical Reasoning Prompt Templates...")

    # Test data
    test_context = "All students attend school. Mary is a student. No one who attends school works full time."
    test_question = "Based on the above information, is the following statement true, false, or uncertain? Mary works full time."
    test_options = ["A) True", "B) False", "C) Uncertain"]
    
    # Test different templates
    prompt_manager = PromptManager()
    
    for template_name in ["direct", "cot", "json_direct", "json_cot"]:
        print(f"\n=== {template_name.upper()} TEMPLATE ===")
        
        if "chat" in template_name:
            messages = prompt_manager.format_for_chat_model(template_name, test_context, test_question, test_options)
            print("System:", messages[0]["content"])
            print("User:", messages[1]["content"][:200] + "...")
        else:
            template = prompt_manager.get_template(template_name)
            prompt = template.create_full_prompt(test_context, test_question, test_options)
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
