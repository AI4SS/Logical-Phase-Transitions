# Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2601.02902v1">
    <img src="https://img.shields.io/badge/arXiv-2601.02902-b31b1b?style=flat-square">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  </a>
</p>

![Pipeline Overview](assets/Figure2_pipeline.png)

This repository contains the official implementation for the paper **"Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning"**. 🧠

> **Abstract**: Symbolic logical reasoning is critical for reliable decision-making 🧠. In this study, we reveal a previously unrecognized phenomenon: **Logical Phase Transitions**. Unlike most capabilities that degrade smoothly, logical reasoning remains stable within a regime but **collapses abruptly** beyond a critical complexity threshold—mirroring physical phase transitions like water freezing. To address this, we propose **Neuro-Symbolic Curriculum Tuning**, a principled framework that adaptively aligns natural language with logical symbols and reshapes training dynamics to progressively strengthen reasoning at increasing logical depths.

---
## ✨ Highlight

Our approach consistently mitigates reasoning collapse at high complexity levels. Compared to original models, **Neuro-Symbolic Curriculum Tuning** yields:
 - 📈 **+1.26** average accuracy gain in **Naive Prompting**. 
 - 📈 **+3.95** average accuracy gain in **Chain-of-Thought (CoT)**. 
 - 🛡️ Improved generalization to unseen logical compositions. 

---
## 🗓️ Timeline

- **[2026-01-08]** We have released the **model weights**.
- **[2026-01-08]** We have released the **code and datasets** covering the entire pipeline.
- **[2026-01-06]** The paper is released on **arXiv**. 

---

## 📂 Project Structure


- **`data-construction/`**: Code to build the NSA-LR dataset from scratch.
- **`dataset/`**: Complete collection of training and testing sets.
- **`main/`**:
    - **`LPT_evaluation/`**: Evaluation suite for detecting Logical Phase Transitions.
    - **`Normal_Evaluation/`**: Standard accuracy evaluation scripts for various datasets.
    - **`vllm-infer/`**: Fast serving configurations.
- **`modelweight/`**: Directory for checking out LoRA adapters.

---

## 🧪 Usage

### 1. Preparation

#### Environment Setup
```bash
pip install -r requirements.txt
```

#### Data Construction (Optional)
The released datasets can be used directly for reproducing the main results. 
The following steps are provided for completeness.If you wish to reproduce the dataset generation process:

```bash
cd data-construction

# 1. Generate Skeletons
python3 logic_skeleton_generator.py --mode easy --num 500 --output_dir outputs/logic_data

# 2. Translate to Language
python3 logic_skeleton_translator.py \
    --model_name gpt-4o --api_key "YOUR_API_KEY" \
    --data_dir outputs/logic_data --num 100 --output_dir outputs/translated_data --mode hard

# 3. Final Problem Generation
python3 fol_problem_generator.py \
    --model_name gpt-4o --api_key "YOUR_API_KEY" \
    --filepath outputs/translated_data/hard-100.json --output_dir outputs/final_data --mode normal_generation
```

### 2. Run

#### Deploy Model
First, download the [\[models\]](https://huggingface.co/NormanandJimmy/Theta_mix) and place them in `modelweight/ours-theta/`.

Then, launch the vLLM server:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules lpt-lora=./modelweight/ours-theta \
    --port 8080
```

#### Running
Perform the main LPT evaluation:
```bash
cd main/LPT_evaluation
python run_evaluation.py evaluate --config configs/evaluation_config.yaml
```

### 3. Evaluation

#### LPT Analysis
Analyze the results to observe phase transitions:
```bash
cd main/LPT_evaluation
python run_evaluation.py analyze --file results/your_results.json
```

#### Normal Accuracy Evaluation
Run standard benchmarks (e.g., NSA-LR, FOLIO, ProntoQA):

```bash
cd main/Normal_Evaluation

# Example: NSA-LR
cd NSA-LR
python nsalr-cot-ollama.py

```

---

## 📖 Citation

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

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
