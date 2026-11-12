"""
benchmark.py
─────────────
Runs a comprehensive benchmark comparing:
  1. The base pretrained model (via HuggingFace)
  2. The fine-tuned model (after QLoRA + merge)
  3. The quantized GGUF model (Q4_K_M via llama.cpp)

Measures:
  • Answer quality (ROUGE-L)
  • Model size (disk usage)
  • Peak RAM usage
  • Inference latency (seconds per response)
  • Throughput (tokens/second)

Results are saved to results/benchmark.csv

What you will learn here:
  • Memory profiling with psutil
  • Comparative benchmarking across model formats
  • Writing structured experiment results to CSV
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import psutil
import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)

# ── A fixed set of benchmark questions ────────────────────────────────────────
# These cover the core evaluation categories defined in the project spec.

BENCHMARK_QUESTIONS = [
    # Space-domain factual
    {
        "category": "factual",
        "instruction": "What is a geostationary orbit and why is it used for communication satellites?",
        "reference": "A geostationary orbit is a circular orbit at approximately 35,786 km altitude above the equator where a satellite's orbital period matches Earth's rotation, keeping it stationary relative to the ground, making it ideal for continuous communication coverage over a fixed area.",
    },
    {
        "category": "factual",
        "instruction": "What is the purpose of a satellite attitude control system?",
        "reference": "An attitude control system maintains or changes the orientation of a spacecraft in space using reaction wheels, magnetorquers, or thrusters to point antennas, solar panels, or scientific instruments in the correct direction.",
    },
    # Subsystem explanation
    {
        "category": "subsystem",
        "instruction": "Explain the purpose of a satellite power subsystem.",
        "reference": "A satellite power subsystem generates, stores, and distributes electrical energy to onboard components, typically using solar panels to generate power and batteries to store energy during eclipse periods.",
    },
    {
        "category": "subsystem",
        "instruction": "What does the thermal control subsystem do on a spacecraft?",
        "reference": "The thermal control subsystem maintains all spacecraft components within their operating temperature ranges using passive methods like coatings and insulation, and active methods like heaters and heat pipes.",
    },
    # Terminology
    {
        "category": "terminology",
        "instruction": "What does GGUF stand for and why is it used in machine learning?",
        "reference": "GGUF (GPT-Generated Unified Format) is a file format for storing language models designed for efficient inference with llama.cpp, replacing the older GGML format with better support for metadata and quantization.",
    },
    {
        "category": "terminology",
        "instruction": "What is delta-v in the context of spacecraft propulsion?",
        "reference": "Delta-v (Δv) is a measure of the change in velocity a spacecraft can achieve using its propulsion system, used to characterise the amount of fuel required for orbital manoeuvres.",
    },
    # Conceptual reasoning
    {
        "category": "reasoning",
        "instruction": "Why do small language models struggle with complex multi-step reasoning compared to large language models?",
        "reference": "Small language models have fewer parameters and limited capacity to store world knowledge and reasoning patterns. Multi-step reasoning requires maintaining intermediate results across many inference steps, which needs more representational capacity than compact models typically have.",
    },
    # Out-of-domain (should handle gracefully)
    {
        "category": "out_of_domain",
        "instruction": "Write a Python function to sort a list of integers.",
        "reference": "This question is outside the space and aerospace domain that SpaceSLM-mini was trained on.",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def get_model_size_mb(model_path: str) -> float:
    """Return total size of a directory or file in megabytes."""
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return 0.0


def get_process_ram_mb() -> float:
    """Return current process RAM usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def compute_rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    except Exception:
        return 0.0


# ── Benchmark runners ──────────────────────────────────────────────────────────

def benchmark_gguf(model_path: str, questions: list[dict]) -> dict:
    """Benchmark the GGUF quantized model."""
    from llama_cpp import Llama

    log.info(f"Loading GGUF model: {model_path}")
    ram_before = get_process_ram_mb()
    llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
    ram_after = get_process_ram_mb()
    peak_ram = ram_after - ram_before

    latencies, rouge_scores, token_counts = [], [], []

    for q in tqdm(questions, desc="GGUF benchmark", unit="q"):
        prompt = build_prompt(q["instruction"])
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=256, temperature=0.1,
                  stop=["### Instruction:"], echo=False)
        latency = time.perf_counter() - t0

        pred      = out["choices"][0]["text"].strip()
        n_tokens  = out["usage"]["completion_tokens"]
        rouge     = compute_rouge_l(pred, q["reference"])

        latencies.append(latency)
        rouge_scores.append(rouge)
        token_counts.append(n_tokens)

    return {
        "model":          "GGUF Q4_K_M",
        "model_size_mb":  get_model_size_mb(model_path),
        "peak_ram_mb":    peak_ram,
        "mean_latency_s": sum(latencies) / len(latencies),
        "tokens_per_sec": sum(token_counts) / sum(latencies),
        "mean_rouge_l":   sum(rouge_scores) / len(rouge_scores),
    }


def benchmark_hf(model_path: str, model_label: str, questions: list[dict]) -> dict:
    """Benchmark a HuggingFace model (base or fine-tuned)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading HF model: {model_path}")
    ram_before = get_process_ram_mb()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    ram_after = get_process_ram_mb()
    peak_ram  = ram_after - ram_before

    latencies, rouge_scores, token_counts = [], [], []

    for q in tqdm(questions, desc=f"{model_label} benchmark", unit="q"):
        prompt = build_prompt(q["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.1,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        latency   = time.perf_counter() - t0
        new_toks  = outputs[0][inputs["input_ids"].shape[1]:]
        pred      = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
        rouge     = compute_rouge_l(pred, q["reference"])

        latencies.append(latency)
        rouge_scores.append(rouge)
        token_counts.append(len(new_toks))

    return {
        "model":          model_label,
        "model_size_mb":  get_model_size_mb(model_path),
        "peak_ram_mb":    peak_ram,
        "mean_latency_s": sum(latencies) / len(latencies),
        "tokens_per_sec": sum(token_counts) / sum(latencies),
        "mean_rouge_l":   sum(rouge_scores) / len(rouge_scores),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = {}
    try:
        with open("configs/training_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        pass

    all_results = []

    # ── GGUF model ────────────────────────────────────────────────────────────
    gguf_path = cfg.get("inference", {}).get("gguf_model_path", "models/gguf/space-slm-q4_k_m.gguf")
    if Path(gguf_path).exists():
        log.info("\n[ Benchmarking GGUF Q4_K_M model ]")
        result = benchmark_gguf(gguf_path, BENCHMARK_QUESTIONS)
        all_results.append(result)
    else:
        log.warning(f"GGUF model not found, skipping: {gguf_path}")

    # ── Fine-tuned HF model ───────────────────────────────────────────────────
    merged_path = "models/merged_fp16"
    if Path(merged_path).exists():
        log.info("\n[ Benchmarking Fine-tuned HF model ]")
        result = benchmark_hf(merged_path, "Fine-tuned FP16", BENCHMARK_QUESTIONS)
        all_results.append(result)

    # ── Base model ────────────────────────────────────────────────────────────
    base_model_id = cfg.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    log.info(f"\n[ Benchmarking Base model: {base_model_id} ]")
    result = benchmark_hf(base_model_id, "Base (pretrained)", BENCHMARK_QUESTIONS)
    all_results.append(result)

    # ── Save results ──────────────────────────────────────────────────────────
    if not all_results:
        log.error("No results to save.")
        return

    output_path = Path("results/benchmark.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    log.info(f"\n✅ Benchmark complete! Results saved: {output_path}")

    # Print summary table
    print("\n" + "═" * 70)
    print(f"{'Model':<25} {'Size (MB)':>10} {'RAM (MB)':>10} {'Latency (s)':>12} {'tok/s':>8} {'ROUGE-L':>9}")
    print("─" * 70)
    for r in all_results:
        print(
            f"{r['model']:<25} {r['model_size_mb']:>10.1f} {r['peak_ram_mb']:>10.1f} "
            f"{r['mean_latency_s']:>12.2f} {r['tokens_per_sec']:>8.1f} {r['mean_rouge_l']:>9.4f}"
        )
    print("═" * 70)


if __name__ == "__main__":
    main()
