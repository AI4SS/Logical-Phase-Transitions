# Data Construction

This directory contains the modified codebase from [ProverGen](https://openreview.net/forum?id=C25SgeXWjE) (Qi et al., 2025), an ICLR 2025 paper on "Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation".

## Overview

We have extended the original ProverGen framework to construct the **Neuro-Symbolic Alignment Dataset for Logical Reasoning (NSA-LR)**. Our modifications enable the construction of datasets with **full-chain matching between Natural Language (NL) and First-Order Logic (FOL)** representations.

For the original ProverGen documentation and usage, please refer to [README_original.md](./README_original.md).

## Dataset: NSA-LR

Building on the data-construction principles of ProverGen (Qi et al., 2025), NSA-LR provides paired natural language (NL) and FOL representations for every sample. All NL propositions, premises, and reasoning steps are translated into explicit predicates, quantifiers, connectives, and multi-step reasoning chains, following the rules defined in the original framework.

Each statement is independently translated by GPT-5 and Qwen3-Max (Yang et al., 2025a); matching outputs undergo CFG (Context-Free Grammar) validation, while mismatches are manually adjudicated.

## Generation Pipeline

The data construction process follows three main steps:

### Step 1: Logic Skeleton Generation

Generate the logical structure using symbolic provers with a novel top-down approach.

```bash
python3 logic_skeleton_generator.py --mode easy --num 500 --output_dir outputs/logic_data
```

**Parameters:**
- `mode`: Difficulty level (`easy`, `medium`, `hard`)
- `num`: Number of logic skeletons to generate
- `output_dir`: Output directory

The script also allows customization of the distribution of answers ([True, False, Uncertain]) and the proportion of composite conclusions.

<details><summary>Advanced Parameters</summary><p>

- `goal_value_probs`: Distribution of [True, False, Uncertain] (e.g., [0.4, 0.3, 0.3])
- `rule_candidate_path`: Path to the rule pool file
- `rule_as_goal_proportion`: Proportion of fact vs. rule conclusions (e.g., [0.7, 0.3])
- `fact_num_threshold`: If the fact pool size surpasses this threshold, there's a chance the fact will be provided directly
- `fact_num_prob`: Probability of directly providing a fact

</p></details>

### Step 2: Logic Skeleton Translation

Convert logic expressions into natural language using LLMs.

```bash
python3 logic_skeleton_translator.py \
    --model_name gpt-5 \
    --base_url your-base-url \
    --api_key "your-api-key" \
    --data_dir outputs/logic_data \
    --num 100 --start 0 --end 100 \
    --output_dir outputs/translated_data \
    --mode hard
```

**Parameters:**
- `model_name`: LLM for translation (e.g., gpt-5, qwen3-max)
- `base_url`/`api_key`: API credentials
- `data_dir`: Path to the logic skeleton files produced in Step 1
- `num`: Total number of logic skeletons
- `start`/`end`: Index range for processing
- `output_dir`: Directory to store the output file
- `mode`: Difficulty level

### Step 3: FOL Problem Generation

Generate complete FOL problems with optional data augmentation.

```bash
python3 fol_problem_generator.py \
    --model_name gpt-5 \
    --base_url your-base-url \
    --api_key "your-api-key" \
    --filepath outputs/translated_data/hard-100-0_100.json \
    --start 0 --end 100 \
    --output_dir outputs/final_data \
    --mode normal_generation
```

**Parameters:**
- `model_name`: LLM for generation
- `base_url`/`api_key`: API credentials
- `filepath`: Path to the translated files produced in Step 2
- `start`/`end`: Index range for processing
- `output_dir`: Directory to store the output file
- `mode`: Generation mode (`normal_generation`, `step_augment`, `uncertain_augment`)

## Citation

If you use this modified codebase or NSA-LR dataset, please cite the original ProverGen paper:

```bibtex
@inproceedings{
qi2025large,
title={Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation},
author={Chengwen Qi and Ren Ma and Bowen Li and He Du and Binyuan Hui and Jinwang Wu and Yuanjun Laili and Conghui He},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=C25SgeXWjE}
}
```
