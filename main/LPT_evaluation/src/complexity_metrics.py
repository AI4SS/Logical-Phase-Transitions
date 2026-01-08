"""
Logical Complexity Metric (LoCM) Module

This module implements the LoCM metric as described in the paper.
The LoCM assigns each reasoning instance a scalar score that captures its logic difficulty.

Definition 1 (LoCM). For a reasoning instance φ expressed in FOL, let P = {p1, ..., pN_φ}
denote the set of premises, where N_φ = |P|. Let O = {∧, ∨, ¬, ⊕, →, ↔, ∀, ∃} denote
the set of logical operators, including Boolean connectives and quantifiers. For each
operator o ∈ O, let freq(o, φ) denote its occurrence count in φ. Let h denote the
number of reasoning hops in the corresponding reasoning chain.

The LoCM is defined as:
    LoCM(φ) = f(Σ_{o∈O} ω(o) * freq(o, φ) + γh(φ))

where:
- ω(o) assigns a symbolic-complexity weight to each operator
- f(·) is a monotonic transformation function used to stabilize scale
- γ is the weight coefficient for hop count
- h(φ) is the number of reasoning hops

The metric LoCM(φ) yields a single scalar score that quantifies the logical difficulty
of a given reasoning instance.
"""

from typing import List
from dataclasses import dataclass
from enum import Enum

class LogicalOperator(Enum):
    """
    Logical operators set O = {∧, ∨, ¬, ⊕, →, ↔, ∀, ∃}

    Including Boolean connectives and quantifiers as defined in LoCM.
    """
    # Basic connectives
    AND = "∧"           # Conjunction
    OR = "∨"            # Disjunction

    # Conditional connectives
    IMPLIES = "→"       # Implication
    IFF = "↔"           # If and only if

    # Negation
    NOT = "¬"           # Negation

    # Other
    XOR = "⊕"           # Exclusive OR

    # Quantifiers
    FORALL = "∀"        # Universal quantifier
    EXISTS = "∃"        # Existential quantifier

@dataclass
class ComplexityWeights:
    """
    Complexity weights configuration for LoCM calculation.

    Attributes:
        basic_connectives: Weight ω for basic connectives (∧, ∨)
        conditional_connectives: Weight ω for conditionals (→, ↔)
        negation: Weight ω for negation (¬)
        xor: Weight ω for XOR (⊕)
        quantifiers: Weight ω for quantifiers (∀, ∃)
        hop_weight: Weight coefficient γ for hop count h(φ)
        transform_function: Monotonic transformation function f(·)
                              Options: 'linear', 'sqrt', 'log'
    """
    # Semantic complexity weights - operator weights ω(o)
    basic_connectives: float = 1.0      # ∧, ∨
    conditional_connectives: float = 3.0  # →, ↔
    negation: float = 2.0              # ¬
    xor: float = 3.5                   # ⊕
    quantifiers: float = 2.0           # ∀, ∃

    # Structural complexity weights
    nesting_multiplier: float = 0      # Nesting depth multiplier
    variable_binding_weight: float = 0  # Variable binding weight

    # Monotonic transformation function f(·): 'linear', 'sqrt', 'log'
    # linear: LoCM = Σ(ω(o) * count(o)) + γ * h
    # sqrt: LoCM = Σ(ω(o) * sqrt(count(o))) + γ * sqrt(h)
    # log: LoCM = Σ(ω(o) * log(count(o) + 1)) + γ * log(h + 1)
    transform_function: str = 'linear'

    # Hop count weight γ (coefficient for h(φ))
    hop_weight: float = 2.0

@dataclass
class ComplexityScore:
    """
    Complexity score for a reasoning instance.

    Attributes:
        semantic: Semantic complexity Σ(ω(o) * f(freq(o, φ)))
        structural: Structural complexity γ * f(h(φ))
        total: Total LoCM score
    """
    structural: float = 0.0    # Structural complexity γ * f(h)
    semantic: float = 0.0      # Semantic complexity Σ(ω(o) * freq(o, φ))
    total: float = 0.0         # Total LoCM score

    def __post_init__(self):
        self.total = self.semantic + self.structural

class ComplexityCalculator:
    """
    LoCM calculator implementing the metric from the paper.

    Computes: LoCM(φ) = f(Σ_{o∈O} ω(o) * freq(o, φ) + γh(φ))
    """

    def __init__(self, weights: ComplexityWeights = None):
        self.weights = weights or ComplexityWeights()

        # Operator weight mapping ω(o)
        self.operator_weights = {
            LogicalOperator.AND: self.weights.basic_connectives,
            LogicalOperator.OR: self.weights.basic_connectives,
            LogicalOperator.IMPLIES: self.weights.conditional_connectives,
            LogicalOperator.IFF: self.weights.conditional_connectives,
            LogicalOperator.NOT: self.weights.negation,
            LogicalOperator.XOR: self.weights.xor,
            LogicalOperator.FORALL: self.weights.quantifiers,
            LogicalOperator.EXISTS: self.weights.quantifiers,
        }

        # Select transformation function f(·)
        import math
        if self.weights.transform_function == 'sqrt':
            self.transform = lambda x: math.sqrt(x) if x > 0 else 0
        elif self.weights.transform_function == 'log':
            self.transform = lambda x: math.log(x + 1)  # log(x+1) to avoid log(0)
        else:  # 'linear' or default
            self.transform = lambda x: x

    def calculate_semantic_complexity(self, operators: List[LogicalOperator]) -> float:
        """
        Calculate semantic complexity: Σ_{o∈O} ω(o) * f(freq(o, φ))

        Args:
            operators: List of logical operators in the expression

        Returns:
            Semantic complexity score
        """
        # Count occurrences of each operator freq(o, φ)
        from collections import Counter
        operator_counts = Counter(operators)

        total_complexity = 0.0
        for op, count in operator_counts.items():
            weight = self.operator_weights.get(op, 0.0)
            # Apply transformation: ω(o) * f(freq(o, φ))
            total_complexity += weight * self.transform(count)

        return total_complexity

    def calculate_structural_complexity(self,
                                      hop_count: int,
                                      num_variables: int = 0,
                                      num_bound_variables: int = 0) -> float:
        """
        Calculate structural complexity: γ * f(h(φ))

        Args:
            hop_count: Number of reasoning hops h(φ)
            num_variables: Number of variables (deprecated, kept for compatibility)
            num_bound_variables: Number of bound variables (deprecated, kept for compatibility)

        Returns:
            Structural complexity score
        """
        # Hop count contribution: γ * f(h)
        hop_score = self.weights.hop_weight * self.transform(hop_count)

        # Variable complexity: bound variables are more complex than free variables
        # Preserve original linear calculation (if weights are 0, no contribution)
        variable_score = (num_variables * self.weights.nesting_multiplier +
                         num_bound_variables * self.weights.variable_binding_weight)

        return hop_score + variable_score

def get_operator_from_symbol(symbol: str) -> LogicalOperator:
    """Map symbol string to logical operator from set O"""
    symbol_map = {
        "∧": LogicalOperator.AND,
        "&": LogicalOperator.AND,
        "and": LogicalOperator.AND,
        "∨": LogicalOperator.OR, 
        "|": LogicalOperator.OR,
        "or": LogicalOperator.OR,
        "→": LogicalOperator.IMPLIES,
        "->": LogicalOperator.IMPLIES,
        "implies": LogicalOperator.IMPLIES,
        "↔": LogicalOperator.IFF,
        "<->": LogicalOperator.IFF,
        "iff": LogicalOperator.IFF,
        "¬": LogicalOperator.NOT,
        "~": LogicalOperator.NOT,
        "not": LogicalOperator.NOT,
        "⊕": LogicalOperator.XOR,
        "xor": LogicalOperator.XOR,
        "∀": LogicalOperator.FORALL,
        "forall": LogicalOperator.FORALL,
        "∃": LogicalOperator.EXISTS,
        "exists": LogicalOperator.EXISTS,
    }
    
    return symbol_map.get(symbol.lower())
