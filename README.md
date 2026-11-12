# 🚀 SpaceSLM-mini

> **Domain-Specific Small Language Model for Space and Aerospace Applications**

SpaceSLM-mini is an experimental domain-adaptation and edge-deployment project focused on fine-tuning a Small Language Model (SLM) for space-domain question answering using **Parameter-Efficient Fine-Tuning (PEFT)** and **QLoRA**.

The project explores how a compact language model can learn space-related terminology and technical concepts from domain-specific datasets while remaining efficient enough for local inference on resource-constrained hardware.

---

## 🎯 Project Objectives

| # | Objective |
|---|-----------|
| 1 | Develop a domain-specific SLM for space and aerospace question answering |
| 2 | Collect, clean and structure domain-specific text datasets from arXiv |
| 3 | Fine-tune a pretrained language model using QLoRA |
| 4 | Reduce model storage and memory requirements through quantization |
| 5 | Deploy the quantized model for local CPU inference |
| 6 | Evaluate model quality, memory consumption and inference performance |
| 7 | Compare pretrained and fine-tuned model behaviour |

---

## 🏗️ System Architecture

```
Space-Domain Data Sources (arXiv API)
              │
              ▼
   Dataset Collection & Cleaning
   (deduplication, QA generation)
              │
              ▼
       Training Dataset
              │
              ▼
   Qwen2.5-0.5B-Instruct (base)
              │
              ▼
  QLoRA Fine-Tuning (Google Colab T4)
              │
              ▼
        LoRA Adapter
              │
              ▼
      Adapter Merging (FP16)
              │
              ▼
       GGUF Conversion
              │
              ▼
   4-bit Quantization (Q4_K_M)
              │
              ▼
  Local Inference (llama.cpp / CPU)
              │
              ▼
    Benchmark & Evaluation
```

---

## 🧠 Model Information

| Component | Technology |
|-----------|-----------|
| Base model | Qwen2.5-0.5B-Instruct |
| Fine-tuning method | QLoRA |
| Fine-tuning framework | Hugging Face PEFT |
| Deep learning | PyTorch |
| Training library | Transformers / TRL |
| Quantization (training) | 4-bit NF4 (BitsAndBytes) |
| Model export format | GGUF |
| Inference runtime | llama.cpp |
| Training environment | Google Colab (NVIDIA T4 GPU) |
| Local hardware | 16 GB RAM, Intel Iris Xe |

---

## 📚 Dataset

### Sources
- **arXiv** – `astro-ph.IM` (Instrumentation & Methods)
- **arXiv** – `astro-ph.EP` (Earth & Planetary)
- **arXiv** – `physics.space-ph` (Space Physics)
- **arXiv** – `eess.SP` (Signal Processing / satellite comms)

### Pipeline
1. Collect research abstracts via arXiv API
2. Normalize and clean extracted content (LaTeX removal, unicode)
3. Remove duplicate and low-quality samples (content hashing)
4. Generate instruction / QA pairs from abstracts
5. Validate and split into train / val / eval sets

### Format (Alpaca-style JSONL)
```json
{
  "instruction": "Explain the purpose of a satellite power subsystem.",
  "input": "",
  "output": "A satellite power subsystem generates, stores and distributes electrical energy to onboard components."
}
```

### Statistics

| Metric | Value |
|--------|-------|
| Raw samples | TBD |
| Processed samples | TBD |
| Training examples | TBD |
| Validation examples | TBD |
| Evaluation examples | TBD |

---

## ⚙️ Fine-Tuning with QLoRA

QLoRA keeps the quantized base model **frozen** and trains only lightweight LoRA adapter matrices — roughly 1% of total parameters.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-0.5B-Instruct |
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 4 (× 4 grad accum = 16 effective) |
| Optimizer | paged_adamw_8bit |
| Training hardware | Google Colab T4 GPU |

> **Hardware note:** No local NVIDIA GPU required. Training is performed on Google Colab's cloud GPU. The local laptop is used for dataset preparation, local inference, and CPU benchmarking.

---

## 📦 Quantization & Edge Deployment

```
Fine-tuned HF Model
        │
        ▼ merge_adapter.py
   FP16 Merged Model
        │
        ▼ convert_gguf.sh (llama.cpp)
   GGUF FP16
        │
        ▼ llama-quantize Q4_K_M
   Q4_K_M GGUF  (~340 MB)
        │
        ▼
  Local CPU Inference
```

---

## 📊 Benchmark Results

> Results will be populated after running experiments.

| Metric | Base Model | Fine-tuned | Quantized (Q4_K_M) |
|--------|-----------|-----------|-------------------|
| Model size | TBD | TBD | TBD |
| Peak RAM | TBD | TBD | TBD |
| Tokens/sec | TBD | TBD | TBD |
| ROUGE-L | TBD | TBD | TBD |
| Avg latency | TBD | TBD | TBD |

---

## 📁 Project Structure

```
SpaceSLM-mini/
│
├── data/
│   ├── raw/                    # Raw arXiv JSONL data
│   ├── processed/              # Cleaned + train/val splits
│   └── evaluation/             # Eval benchmark set
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_qlora_finetuning.ipynb    ← run on Google Colab
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── collect_arxiv.py    # arXiv API data collection
│   │   ├── clean_dataset.py    # Text cleaning & dedup
│   │   └── generate_qa.py      # Instruction QA generation
│   │
│   ├── training/
│   │   ├── train_qlora.py      # QLoRA fine-tuning script
│   │   └── merge_adapter.py    # Merge LoRA → FP16 model
│   │
│   ├── inference/
│   │   └── generate.py         # Local CPU inference (GGUF)
│   │
│   ├── quantization/
│   │   └── convert_gguf.sh     # GGUF conversion pipeline
│   │
│   └── evaluation/
│       ├── evaluate.py         # ROUGE-L evaluation
│       └── benchmark.py        # Full model benchmark
│
├── models/
│   ├── adapter/                # Saved LoRA adapter
│   └── gguf/                   # Quantized GGUF model
│
├── results/
│   ├── metrics.json
│   ├── benchmark.csv
│   └── evaluation_report.md
│
├── configs/
│   └── training_config.yaml    # All hyperparameters
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/hdq4ch/SpaceSLM.git
cd SpaceSLM
```

### 2. Create a Python environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset
```bash
python src/data/collect_arxiv.py
python src/data/clean_dataset.py
python src/data/generate_qa.py
```

### 5. Fine-tune on Google Colab
Open [`notebooks/03_qlora_finetuning.ipynb`](notebooks/03_qlora_finetuning.ipynb) in Google Colab:
- `Runtime → Change runtime type → T4 GPU`
- Run all cells in order
- Download `models/merged_fp16/` when done

### 6. Convert and quantize
```bash
# Requires llama.cpp installed – see convert_gguf.sh for setup instructions
bash src/quantization/convert_gguf.sh
```

### 7. Run local inference
```bash
# Interactive mode
python src/inference/generate.py

# Single question
python src/inference/generate.py -q "What is the purpose of a satellite power subsystem?"

# Or directly with llama.cpp CLI
llama-cli -m models/gguf/space-slm-q4_k_m.gguf \
  -p "### Instruction:\nExplain geostationary orbit.\n\n### Response:\n"
```

### 8. Evaluate and benchmark
```bash
python src/evaluation/evaluate.py --model-type gguf
python src/evaluation/benchmark.py
```

---

## 🔬 Future Improvements

- [ ] Expand dataset with verified aerospace engineering documents
- [ ] Experiment with different LoRA ranks (r=4, 16, 32)
- [ ] Compare multiple compact model architectures (SmolLM, Phi-3-mini)
- [ ] Implement RAG for retrieval from technical engineering manuals
- [ ] Evaluate INT8 and INT4 GGUF quantization levels
- [ ] Explore ONNX Runtime for broader deployment targets
- [ ] Add structured output validation
- [ ] Develop reliability benchmarks for safety-critical assessments
- [ ] Investigate model distillation and structured pruning

---

## ⚠️ Limitations

- Fine-tuning quality depends heavily on dataset quality and coverage
- Research abstracts do not provide complete aerospace engineering specifications
- Quantization can affect answer quality
- Small language models may struggle with complex multi-step reasoning
- **Model outputs must NOT be treated as verified engineering specifications**
- Further testing is required before deployment in safety-critical environments

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.40-yellow)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-orange)

---

## 👨‍💻 Author

**Hadiq C**  
Computer Science Undergraduate | AI/ML & MLOps Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-2022bcs0044--hadiq-black)](https://github.com/2022bcs0044-hadiq)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-hadiqc-blue)](https://www.linkedin.com/in/hadiqc/)

---

> *SpaceSLM-mini is an experimental learning and engineering project exploring domain-specific language model adaptation and efficient local inference for space-domain applications.*
