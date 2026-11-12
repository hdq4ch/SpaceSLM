"""
evaluate.py
────────────
Evaluates the fine-tuned model against a space-domain QA benchmark.

Supports two evaluation modes:
  1. HuggingFace model (base or fine-tuned FP16) – for Colab / GPU evaluation
  2. GGUF model via llama-cpp-python – for local CPU evaluation

Metrics computed:
  • ROUGE-L  : overlap between generated and reference answers
  • Exact match rate
  • Average answer length

What you will learn here:
  • Running batch inference for evaluation
  • ROUGE scoring with the rouge-score library
  • Structuring a reusable evaluation pipeline
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── ROUGE scoring ──────────────────────────────────────────────────────────────

def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between a prediction and a reference string.

    ROUGE-L measures the longest common subsequence (LCS) between strings.
    Score ranges from 0 (no overlap) to 1 (perfect match).
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        log.warning("rouge-score not installed. Run: pip install rouge-score")
        return 0.0


def exact_match(prediction: str, reference: str) -> bool:
    """Case-insensitive exact match after whitespace normalisation."""
    return " ".join(prediction.lower().split()) == " ".join(reference.lower().split())


# ── Inference backends ─────────────────────────────────────────────────────────

def build_prompt(instruction: str, input_text: str = "") -> str:
    """Alpaca-style prompt (must match training template)."""
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,   # low for evaluation (more deterministic)
) -> str:
    """Generate a response using a HuggingFace model."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens (skip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_gguf(llm, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response using a llama-cpp-python Llama object."""
    output = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.1,
        stop=["### Instruction:", "### Input:"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(
    eval_file: str,
    model_type: str,         # "hf" or "gguf"
    model_path: str,
    output_file: str = "results/evaluation_report.md",
    max_samples: Optional[int] = None,
) -> dict:
    """
    Run evaluation and return a metrics dictionary.

    Parameters
    ----------
    eval_file   : path to evaluation JSONL
    model_type  : "hf" (HuggingFace) or "gguf" (llama.cpp)
    model_path  : model directory (hf) or .gguf path
    output_file : where to write the markdown report
    max_samples : limit samples for quick testing
    """
    # Load eval dataset
    samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    log.info(f"Evaluating {len(samples)} samples using [{model_type}] model")

    # Load model
    if model_type == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        generate_fn = lambda prompt: generate_hf(model, tokenizer, prompt)

    elif model_type == "gguf":
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
        generate_fn = lambda prompt: generate_gguf(llm, prompt)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'hf' or 'gguf'.")

    # Evaluation loop
    results = []
    rouge_scores, exact_matches, latencies = [], [], []

    for sample in tqdm(samples, desc="Evaluating", unit="sample"):
        prompt    = build_prompt(sample["instruction"], sample.get("input", ""))
        reference = sample["output"]

        t0 = time.perf_counter()
        prediction = generate_fn(prompt)
        latency    = time.perf_counter() - t0

        rouge  = compute_rouge_l(prediction, reference)
        em     = exact_match(prediction, reference)

        rouge_scores.append(rouge)
        exact_matches.append(int(em))
        latencies.append(latency)

        results.append({
            "instruction": sample["instruction"],
            "reference":   reference,
            "prediction":  prediction,
            "rouge_l":     rouge,
            "exact_match": em,
            "latency_s":   latency,
        })

    # Aggregate metrics
    metrics = {
        "num_samples":     len(samples),
        "mean_rouge_l":    sum(rouge_scores) / len(rouge_scores),
        "exact_match_rate": sum(exact_matches) / len(exact_matches),
        "mean_latency_s":  sum(latencies) / len(latencies),
        "total_time_s":    sum(latencies),
    }

    _save_report(results, metrics, output_file, model_type, model_path)
    return metrics


# ── Report generation ──────────────────────────────────────────────────────────

def _save_report(
    results: list[dict],
    metrics: dict,
    output_file: str,
    model_type: str,
    model_path: str,
) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SpaceSLM-mini Evaluation Report\n",
        f"**Model type:** {model_type}  \n",
        f"**Model path:** `{model_path}`  \n\n",
        "## Summary Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Samples evaluated | {metrics['num_samples']} |",
        f"| Mean ROUGE-L | {metrics['mean_rouge_l']:.4f} |",
        f"| Exact Match Rate | {metrics['exact_match_rate']:.2%} |",
        f"| Mean Latency | {metrics['mean_latency_s']:.2f} s |",
        f"| Total Time | {metrics['total_time_s']:.1f} s |",
        "\n## Sample Results (first 10)\n",
    ]

    for i, r in enumerate(results[:10], 1):
        lines += [
            f"### Sample {i}",
            f"**Instruction:** {r['instruction']}",
            f"**Reference:** {r['reference'][:200]}...",
            f"**Prediction:** {r['prediction'][:200]}...",
            f"**ROUGE-L:** {r['rouge_l']:.4f} | **Exact Match:** {r['exact_match']}",
            "",
        ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save raw metrics as JSON
    metrics_path = Path(output_file).parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Report saved: {output_file}")
    log.info(f"Metrics saved: {metrics_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SpaceSLM model")
    parser.add_argument("--model-type", choices=["hf", "gguf"], default="gguf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--eval-file",  default="data/evaluation/eval.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    model_path = args.model_path or cfg["inference"]["gguf_model_path"]

    metrics = evaluate(
        eval_file=args.eval_file,
        model_type=args.model_type,
        model_path=model_path,
        max_samples=args.max_samples,
    )

    print("\n── Evaluation Results ─────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
