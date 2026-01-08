"""
Logical Complexity Analyzer for LoCM Calculation

This module implements the Logical Complexity Metric (LoCM) analyzer that:
1. Parses First-Order Logic (FOL) expressions
2. Extracts operators from the set O = {∧, ∨, ¬, ⊕, →, ↔, ∀, ∃}
3. Calculates operator frequencies freq(o, φ)
4. Computes reasoning hop count h(φ)
5. Applies LoCM formula: LoCM(φ) = f(Σ_{o∈O} ω(o) * freq(o, φ) + γh(φ))

Supports extracting complexity from premises and conclusions in ProverQA and NSA-LR datasets.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from complexity_metrics import (
    LogicalOperator, ComplexityWeights, ComplexityScore,
    ComplexityCalculator, get_operator_from_symbol
)

@dataclass
class FOLAnalysis:
    """
    Analysis result of a FOL expression φ.

    Attributes:
        operators: List of operators extracted from φ
        nesting_depth: Maximum syntactic nesting depth d
        num_variables: Total number of variables
        num_bound_variables: Number of bound variables (quantified)
        variable_names: List of all variable names
        bound_variables: List of bound variable names
    """
    operators: List[LogicalOperator]
    nesting_depth: int
    num_variables: int
    num_bound_variables: int
    variable_names: List[str]
    bound_variables: List[str]

@dataclass
class FOLIOComplexity:
    """
    Complete LoCM analysis for a reasoning sample.

    Attributes:
        premises_complexity: LoCM score for premises
        conclusion_complexity: LoCM score for conclusion
        total_complexity: Combined LoCM score for the instance
        premises_analysis: FOL analysis of premises
        conclusion_analysis: FOL analysis of conclusion
    """
    premises_complexity: ComplexityScore
    conclusion_complexity: ComplexityScore
    total_complexity: ComplexityScore
    premises_analysis: FOLAnalysis
    conclusion_analysis: FOLAnalysis

class FOLParser:
    """
    First-Order Logic expression parser.

    Extracts operators from O = {∧, ∨, ¬, ⊕, →, ↔, ∀, ∃} and computes
    syntactic features for LoCM calculation.
    """

    def __init__(self):
        # Regex patterns for logical operators from set O
        self.operator_patterns = {
            r'∧|&|and': LogicalOperator.AND,
            r'∨|\||or': LogicalOperator.OR,
            r'→|->|implies': LogicalOperator.IMPLIES,
            r'↔|<->|iff': LogicalOperator.IFF,
            r'¬|~|not': LogicalOperator.NOT,
            r'⊕|xor': LogicalOperator.XOR,
            r'∀|forall': LogicalOperator.FORALL,
            r'∃|exists': LogicalOperator.EXISTS,
        }

        # Variable pattern: typically single letter or letter+number
        self.variable_pattern = r'\b[a-z][a-z0-9]*\b'

        # Quantifier binding pattern: ∀x, ∃y, etc.
        self.quantifier_pattern = r'(∀|∃|forall|exists)\s*([a-z][a-z0-9]*)'

    def extract_operators(self, fol_expr: str) -> List[LogicalOperator]:
        """
        Extract all operators o ∈ O from expression φ and calculate freq(o, φ).

        Args:
            fol_expr: FOL expression φ

        Returns:
            List of operators with their frequencies
        """
        operators = []

        # Convert expression to lowercase for matching
        expr_lower = fol_expr.lower()

        for pattern, operator in self.operator_patterns.items():
            matches = re.findall(pattern, expr_lower)
            operators.extend([operator] * len(matches))

        return operators

    def calculate_nesting_depth(self, fol_expr: str) -> int:
        """
        Calculate maximum syntactic nesting depth d of expression φ.

        Args:
            fol_expr: FOL expression φ

        Returns:
            Maximum nesting depth d
        """
        max_depth = 0
        current_depth = 0

        for char in fol_expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        return max_depth

    def extract_variables(self, fol_expr: str) -> Tuple[List[str], List[str]]:
        """
        Extract all variables and bound variables from φ.

        Args:
            fol_expr: FOL expression φ

        Returns:
            Tuple of (all_variables, bound_variables)
        """
        # Extract all variables
        all_variables = re.findall(self.variable_pattern, fol_expr.lower())

        # Extract bound variables (variables after quantifiers)
        bound_matches = re.findall(self.quantifier_pattern, fol_expr.lower())
        bound_variables = [match[1] for match in bound_matches]

        # Remove duplicates
        all_variables = list(set(all_variables))
        bound_variables = list(set(bound_variables))

        return all_variables, bound_variables

    def estimate_chain_length(self, fol_expr: str) -> int:
        """Estimate reasoning chain length (based on connector count) - Deprecated"""
        # Keep method for backward compatibility but not used
        operators = self.extract_operators(fol_expr)
        reasoning_operators = [op for op in operators if op != LogicalOperator.NOT]
        return len(reasoning_operators) + 1

    def detect_indirect_reasoning(self, fol_expr: str) -> bool:
        """Detect if expression contains indirect reasoning pattern - Deprecated"""
        # Keep method for backward compatibility but not used
        has_negation = LogicalOperator.NOT in self.extract_operators(fol_expr)
        has_implication = LogicalOperator.IMPLIES in self.extract_operators(fol_expr)
        return has_negation and has_implication

    def parse(self, fol_expr: str) -> FOLAnalysis:
        """
        Parse FOL expression φ and return complete analysis.

        Args:
            fol_expr: FOL expression φ

        Returns:
            FOLAnalysis containing operators, nesting depth, and variables
        """
        if not fol_expr or fol_expr.strip() == "":
            return FOLAnalysis([], 0, 0, 0, [], [])

        operators = self.extract_operators(fol_expr)
        nesting_depth = self.calculate_nesting_depth(fol_expr)
        variables, bound_variables = self.extract_variables(fol_expr)

        return FOLAnalysis(
            operators=operators,
            nesting_depth=nesting_depth,
            num_variables=len(variables),
            num_bound_variables=len(bound_variables),
            variable_names=variables,
            bound_variables=bound_variables
        )

class ComplexityAnalyzer:
    """
    Main LoCM analyzer.

    Computes LoCM(φ) = f(Σ_{o∈O} ω(o) * freq(o, φ) + γh(φ)) for reasoning instances.
    """

    def __init__(self, weights: ComplexityWeights = None):
        self.parser = FOLParser()
        self.calculator = ComplexityCalculator(weights)

    def analyze_fol_expression(self, fol_expr: str, hop_count: int = 0) -> Tuple[FOLAnalysis, ComplexityScore]:
        """
        Analyze complexity of a single FOL expression φ.

        Args:
            fol_expr: FOL expression φ
            hop_count: Reasoning hop count h(φ), defaults to 0 which uses FOL nesting depth

        Returns:
            (FOLAnalysis, ComplexityScore) where ComplexityScore contains LoCM(φ)
        """
        analysis = self.parser.parse(fol_expr)

        # Calculate semantic complexity: Σ ω(o) * f(freq(o, φ))
        semantic = self.calculator.calculate_semantic_complexity(analysis.operators)

        # Calculate structural complexity: γ * f(h(φ))
        # Use provided hop_count, or fall back to FOL nesting depth for backward compatibility
        actual_hop_count = hop_count if hop_count > 0 else analysis.nesting_depth
        structural = self.calculator.calculate_structural_complexity(
            hop_count=actual_hop_count,
            num_variables=analysis.num_variables,
            num_bound_variables=analysis.num_bound_variables
        )

        complexity = ComplexityScore(
            semantic=semantic,
            structural=structural
        )

        return analysis, complexity

    def analyze_folio_sample(self, sample: Dict[str, Any], hop_count: int = 0) -> FOLIOComplexity:
        """
        Analyze a reasoning sample using LoCM.

        Args:
            sample: Sample data with premises and conclusion
            hop_count: Reasoning hop count h(φ), defaults to 0 which uses FOL nesting depth

        Returns:
            FOLIOComplexity containing LoCM scores for premises, conclusion, and total
        """
        # Extract FOL expressions for premises and conclusion
        premises_fol = sample.get('premises-FOL', '')
        conclusion_fol = sample.get('conclusion-FOL', '')

        # If list, join into string
        if isinstance(premises_fol, list):
            premises_fol = ' ∧ '.join(premises_fol)
        if isinstance(conclusion_fol, list):
            conclusion_fol = ' ∧ '.join(conclusion_fol)

        # Analyze premises and conclusion
        # Use provided hop_count for premises (reasoning steps h(φ))
        premises_analysis, premises_complexity = self.analyze_fol_expression(premises_fol, hop_count)
        # Don't calculate structural complexity for conclusion (only semantic)
        conclusion_analysis, conclusion_complexity = self.analyze_fol_expression(conclusion_fol, 0)

        # Calculate total complexity
        # Note: Structural complexity only comes from premises (contains hop count info h(φ))
        # Conclusion's structural complexity is ignored as hop count is already reflected in premises
        total_complexity = ComplexityScore(
            semantic=premises_complexity.semantic + conclusion_complexity.semantic,
            structural=premises_complexity.structural  # Only use premises' structural complexity
        )

        return FOLIOComplexity(
            premises_complexity=premises_complexity,
            conclusion_complexity=conclusion_complexity,
            total_complexity=total_complexity,
            premises_analysis=premises_analysis,
            conclusion_analysis=conclusion_analysis
        )

    def batch_analyze_folio(self, folio_data: List[Dict[str, Any]]) -> List[FOLIOComplexity]:
        """
        Batch analyze dataset using LoCM.

        Args:
            folio_data: List of reasoning samples

        Returns:
            List of FOLIOComplexity results
        """
        results = []

        for sample in folio_data:
            try:
                complexity = self.analyze_folio_sample(sample)
                results.append(complexity)
            except Exception as e:
                print(f"Error analyzing sample {sample.get('example_id', 'unknown')}: {e}")
                # Create empty complexity result
                empty_analysis = FOLAnalysis([], 0, 0, 0, [], [])
                empty_complexity = ComplexityScore()
                results.append(FOLIOComplexity(
                    empty_complexity, empty_complexity, empty_complexity,
                    empty_analysis, empty_analysis
                ))

        return results

def load_folio_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset (supports both JSON and JSONL formats).

    Args:
        file_path: Path to data file

    Returns:
        List of data samples
    """
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                # JSONL format: one JSON object per line
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                # JSON format
                data = json.load(f)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

    return data

# Test function
def test_complexity_analyzer():
    """Test LoCM analyzer with sample expressions"""
    analyzer = ComplexityAnalyzer()

    # Test cases
    test_cases = [
        "∀x (P(x) → Q(x))",  # Simple implication
        "∀x ∃y (P(x) ∧ Q(y))",  # Nested quantifiers
        "¬(∀x (P(x) → ¬Q(x)))",  # Complex negation
        "(P ∧ Q) ∨ (R → S)",  # Multiple connectors
    ]

    for i, expr in enumerate(test_cases):
        print(f"\nTest case {i+1}: {expr}")
        analysis, complexity = analyzer.analyze_fol_expression(expr)

        print(f"Operators: {[op.value for op in analysis.operators]}")
        print(f"Nesting depth: {analysis.nesting_depth}")
        print(f"Variables: {analysis.num_variables}")
        print(f"Bound variables: {analysis.num_bound_variables}")
        print(f"LoCM - Semantic: {complexity.semantic:.2f}, Structural: {complexity.structural:.2f}, Total: {complexity.total:.2f}")

if __name__ == "__main__":
    test_complexity_analyzer()
