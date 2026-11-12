"""
merge_adapter.py
─────────────────
Merges the trained LoRA adapter back into the full base model weights,
producing a standard Hugging Face model that can be converted to GGUF.

Why merge?
  During training we have: base_model (frozen, 4-bit) + LoRA adapter (fp32)
  For GGUF conversion we need a single, clean fp16 model with merged weights.

What you will learn here:
  • Loading PEFT adapters with PeftModel
  • Merging LoRA deltas: W_final = W_base + (alpha/r) * B * A
  • Saving a standard HuggingFace model for downstream conversion
"""

import logging
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_and_save(
    base_model_id: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """
    Load the base model in full precision, merge the LoRA adapter,
    and save the merged model.

    Parameters
    ----------
    base_model_id : HuggingFace model hub ID (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
    adapter_path  : path to saved PEFT adapter directory
    output_path   : where to save the merged FP16 model
    """
    log.info(f"Loading base model in FP16: {base_model_id}")
    # Load in float16 without quantization for clean merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    log.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    log.info("Merging LoRA weights into base model...")
    # merge_and_unload() folds A*B into W, then removes the adapter modules
    model = model.merge_and_unload()

    log.info(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer alongside the model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    log.info("✅ Adapter merged and model saved successfully!")
    log.info(f"   Output: {Path(output_path).resolve()}")
    log.info("   Next step: run convert_gguf.sh to convert to GGUF format.")


if __name__ == "__main__":
    cfg = load_config()
    merge_and_save(
        base_model_id=cfg["model"]["base_model_id"],
        adapter_path=cfg["training"]["output_dir"],
        output_path="models/merged_fp16",
    )
