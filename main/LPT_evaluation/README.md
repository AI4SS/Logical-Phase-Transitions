# Logical Phase Transition (LPT) Evaluation Framework

This framework implements the **Logical Complexity Metric (LoCM)** for detecting and analyzing **Logical Phase Transitions (LPT)** in Large Language Models' formal reasoning capabilities.

## Overview

Using LoCM, we evaluate LLMs and observe a previously unrecognized phenomenon: rather than degrading smoothly, reasoning accuracy remains relatively stable over certain complexity ranges and then drops abruptly within specific regions as LoCM increases. This collapse behavior consistently appears across both open- and closed-source LLMs.

Analogous to phase transitions in physics where a system exhibits macroscopic changes once a control variable enters a critical region, we term this phenomenon a **Logical Phase Transition (LPT)** and formalize it as a collapse in model performance governed by logical complexity. The transition occurs over one or more critical intervals **Iₖ = [τₖᵐⁱⁿ, τₖᵐᵃˣ]**, within which accuracy drops sharply as LoCM(φ) enters the interval and stabilizes again once LoCM(φ) exceeds its upper bound.

The discovery of LPTs indicates that direct exposure to high-complexity samples is ineffective, motivating curriculum learning that organizes samples from easier to harder ones. By progressively increasing logical complexity, curriculum learning enables stable traversal of transition regions beyond the LPT regimes.

## Mathematical Definition

### Definition 1 (LoCM)

For a reasoning instance φ expressed in FOL, let P = {p₁, ..., pN_φ} denote the set of premises, where N_φ = |P|. Let **O = {∧, ∨, ¬, ⊕, →, ↔, ∀, ∃}** denote the set of logical operators, including Boolean connectives and quantifiers.

For each operator o ∈ O, let **freq(o, φ)** denote its occurrence count in φ. Let **h** denote the number of reasoning hops.

The **LoCM** is defined as:

```
LoCM(φ) = f( Σ_{o∈O} ω(o) * freq(o, φ) + γh(φ) )
```

Where:
- **ω(o)** assigns a symbolic-complexity weight to each operator
- **f(·)** is a monotonic transformation function (sqrt in this implementation)
- **γ** is the weight coefficient for hop count
- **h(φ)** is the number of reasoning hops

### Implementation Note

This implementation applies **linear aggregation** first, then applies **sqrt transformation** to stabilize the scale. The complexity calculation in [`complexity_metrics.py`](src/complexity_metrics.py) uses `transform_function='linear'`, and the overall sqrt transformation is applied during result analysis in [`run_evaluation.py`](run_evaluation.py).

## Installation

```bash
pip install -r requirements.txt
cd LPT_evaluation
```

## Quick Start

### 1. Run Evaluation

```bash
python run_evaluation.py evaluate --config <config_file>
```

### 2. Analyze Results

```bash
python run_evaluation.py analyze --file <results_file>.json
```

The analyze command automatically applies `sqrt()` transformation to `complexity_score` fields and generates:
- `analysis.txt`: Full analysis with complexity-binned accuracy
- `analysis_woerror.txt`: Error-filtered analysis
- `complexity_plot.png`: Complexity-accuracy correlation plot

## Complexity Weights

| Operator Type | Symbol | Weight ω(o) |
|--------------|--------|------------|
| Basic Connectives | ∧, ∨ | 1.0 |
| Conditional Connectives | →, ↔ | 3.0 |
| Negation | ¬ | 2.0 |
| XOR | ⊕ | 3.5 |
| Quantifiers | ∀, ∃ | 2.0 |

These weights can be customized in [`src/complexity_metrics.py`](src/complexity_metrics.py).

## Datasets

- **ProverQA**: Logical reasoning questions in First-Order Logic format
- **NSA-LR**: Natural Stories with Abstract Reasoning dataset

## Output Format

### Example Result Entry

```json
{
  "example_id": "example_001",
  "context": "All humans are mortal. Socrates is a human.",
  "question": "Is Socrates mortal?",
  "ground_truth": "True",
  "predicted_answer": "True",
  "is_correct": true,
  "complexity_score": 1,
  "response_time": 2.3
}
```

## Computing LoCM

```python
from src.complexity_analyzer import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()

# Analyze a single FOL expression
fol_expr = "∀x (P(x) → Q(x))"
analysis, complexity = analyzer.analyze_fol_expression(fol_expr)

print(f"LoCM Score: {complexity.total:.2f}")
print(f"  - Semantic: {complexity.semantic:.2f}")
print(f"  - Structural: {complexity.structural:.2f}")
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{zhang2026logicalphasetransitionsunderstanding,
      title={Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning}, 
      author={Xinglang Zhang and Yunyao Zhang and ZeLiang Chen and Junqing Yu and Wei Yang and Zikai Song},
      year={2026},
      eprint={2601.02902},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.02902}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ProverQA dataset creators
- NSA-LR dataset creators
- The broader logical reasoning and formal verification community
