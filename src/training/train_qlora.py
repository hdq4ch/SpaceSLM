"""
train_qlora.py
───────────────
Fine-tunes Qwen2.5-0.5B-Instruct using QLoRA on the prepared space-domain
instruction dataset.

Design decisions explained:
  • 4-bit NF4 quantization (BitsAndBytes) → reduces GPU VRAM from ~1 GB to ~0.3 GB
  • LoRA adapters only update ~1% of parameters → fast training, cheap storage
  • SFTTrainer (from TRL) handles padding, packing, and the training loop
  • gradient_checkpointing trades compute for VRAM savings

Run this notebook on Google Colab with a T4 GPU runtime.
"""

import json
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

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


# ── GPU check ──────────────────────────────────────────────────────────────────

def check_gpu() -> None:
    """Print GPU info so you know what Colab gave you."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"✅ GPU detected: {gpu}  ({vram:.1f} GB VRAM)")
    else:
        log.warning("⚠️  No GPU detected – training will be very slow on CPU.")


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_datasets(cfg: dict) -> tuple[Dataset, Dataset]:
    """Load train and validation splits as HuggingFace Dataset objects."""
    train_records = load_jsonl(cfg["dataset"]["train_file"])
    val_records   = load_jsonl(cfg["dataset"]["val_file"])

    log.info(f"Train examples : {len(train_records)}")
    log.info(f"Val examples   : {len(val_records)}")

    return (
        Dataset.from_list(train_records),
        Dataset.from_list(val_records),
    )


# ── Model & tokenizer ──────────────────────────────────────────────────────────

def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",   # required for SFT with causal LMs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_model(model_id: str, quant_cfg: dict) -> AutoModelForCausalLM:
    """
    Load the base model in 4-bit NF4.

    BitsAndBytesConfig tells transformers to:
      • Load weights as 4-bit (not 8-bit or fp32)
      • Use NF4 data type (better than FP4 for language models)
      • Compute in float16 (required for Colab T4)
      • Use double quantization (quantize the quantization constants too)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",           # auto-places layers on GPU / CPU
        trust_remote_code=True,
    )
    return model


# ── LoRA ───────────────────────────────────────────────────────────────────────

def apply_lora(model: AutoModelForCausalLM, lora_cfg: dict) -> AutoModelForCausalLM:
    """
    Freeze the quantized base model and attach trainable LoRA matrices.

    LoRA adds two small matrices (A and B) to selected weight matrices.
    Instead of updating W (huge), we train ΔW = A × B (tiny).
    After training: W_fine = W_pretrained + ΔW
    """
    # Prepare model: cast normalization layers to float32 for stability
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # shows how few params are trained
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    check_gpu()

    model_id  = cfg["model"]["base_model_id"]
    train_cfg = cfg["training"]
    lora_cfg  = cfg["lora"]
    quant_cfg = cfg["quantization"]

    log.info(f"Loading tokenizer: {model_id}")
    tokenizer = load_tokenizer(model_id)

    log.info(f"Loading base model in 4-bit: {model_id}")
    model = load_quantized_model(model_id, quant_cfg)

    log.info("Applying LoRA adapters...")
    model = apply_lora(model, lora_cfg)

    log.info("Loading datasets...")
    train_dataset, val_dataset = load_datasets(cfg)

    # Training arguments (mirrors training_config.yaml)
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",    # the formatted text field from generate_qa.py
        max_seq_length=cfg["model"]["max_seq_length"],
    )

    log.info("🚀 Starting QLoRA training...")
    trainer.train()

    log.info(f"✅ Training complete! Saving adapter to: {train_cfg['output_dir']}")
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
