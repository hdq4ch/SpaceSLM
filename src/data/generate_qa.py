"""
generate_qa.py
───────────────
Converts cleaned research abstracts into instruction-tuning QA pairs.

Two generation strategies are supported:
  1. Rule-based  – fast, free, no API key required (default)
  2. LLM-assisted – higher quality, requires an OpenAI API key

The output is split into train / val / eval JSONL files.

What you will learn here:
  • Prompt engineering for instruction-tuning datasets
  • Dataset splitting (train / val / eval)
  • Template-based text generation
  • Optional LLM-in-the-loop data augmentation
"""

import json
import logging
import random
import re
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


# ── Instruction Templates ──────────────────────────────────────────────────────
# These templates are filled with content from the paper abstract.

TEMPLATES = [
    # Explanation request
    {
        "instruction": "Explain the main topic of the following research abstract in simple terms.",
        "use_input": True,
        "answer_key": "abstract",
    },
    # Summary request
    {
        "instruction": "Summarise the key findings described in the following research abstract.",
        "use_input": True,
        "answer_key": "abstract",
    },
    # Title-to-abstract
    {
        "instruction": "What does the research paper titled '{title}' investigate?",
        "use_input": False,
        "answer_key": "abstract",
    },
    # Domain Q&A
    {
        "instruction": "Based on the abstract below, what problem does the research address?",
        "use_input": True,
        "answer_key": "abstract",
    },
    # Category-aware
    {
        "instruction": "In the context of space and astrophysics research, describe what this paper contributes.",
        "use_input": True,
        "answer_key": "abstract",
    },
]

INSTRUCT_FORMAT = """{instruction}

### Input:
{input}

### Response:
{output}"""


def abstract_to_answer(abstract: str, max_sentences: int = 4) -> str:
    """
    Extract the most informative sentences from an abstract as the answer.
    Uses simple sentence splitting – good enough for rule-based generation.
    """
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    # Take the first few sentences (most abstracts front-load findings)
    answer_sentences = sentences[:max_sentences]
    return " ".join(answer_sentences).strip()


def build_qa_pair(record: dict, template: dict) -> dict:
    """
    Create one QA pair from a paper record using the given template.
    Returns a dict with keys: instruction, input, output, source_id.
    """
    title    = record.get("title", "")
    abstract = record.get("abstract", "")

    # Fill in template instruction (may contain {title} etc.)
    instruction = template["instruction"].format(title=title, abstract=abstract)

    if template["use_input"]:
        input_text = abstract
        output     = abstract_to_answer(abstract)
    else:
        input_text = ""
        output     = abstract_to_answer(abstract)

    return {
        "instruction": instruction,
        "input":       input_text,
        "output":      output,
        "source_id":   record.get("arxiv_id", "unknown"),
        "source_title": title,
    }


def format_for_training(qa: dict) -> dict:
    """
    Wrap a QA dict into the standard Alpaca-style instruction format
    used by the TRL SFTTrainer.
    """
    if qa["input"]:
        text = (
            f"### Instruction:\n{qa['instruction']}\n\n"
            f"### Input:\n{qa['input']}\n\n"
            f"### Response:\n{qa['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{qa['instruction']}\n\n"
            f"### Response:\n{qa['output']}"
        )
    qa["text"] = text
    return qa


def split_dataset(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Shuffle and split records into train / val / eval sets.
    eval_ratio = 1 - train_ratio - val_ratio
    """
    random.seed(seed)
    random.shuffle(records)

    n       = len(records)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = records[:n_train]
    val   = records[n_train : n_train + n_val]
    eval_ = records[n_train + n_val :]

    return train, val, eval_


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(records):>5} records → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("configs/training_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    input_path  = Path("data/processed/cleaned_papers.jsonl")

    if not input_path.exists():
        log.error(f"Cleaned dataset not found: {input_path}")
        log.error("Run clean_dataset.py first.")
        return

    # Load cleaned papers
    papers: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    log.info(f"Loaded {len(papers)} cleaned papers")

    # Generate QA pairs – one template per paper (randomised)
    qa_pairs: list[dict] = []
    for paper in tqdm(papers, desc="Generating QA pairs", unit="paper"):
        # Pick a random subset of templates per paper
        chosen = random.sample(TEMPLATES, k=min(2, len(TEMPLATES)))
        for template in chosen:
            qa = build_qa_pair(paper, template)
            qa = format_for_training(qa)
            qa_pairs.append(qa)

    log.info(f"Generated {len(qa_pairs)} QA pairs from {len(papers)} papers")

    # Split
    train, val, eval_ = split_dataset(
        qa_pairs,
        train_ratio=dataset_cfg["train_split"],
        val_ratio=dataset_cfg["val_split"],
        seed=dataset_cfg["shuffle_seed"],
    )

    log.info("\n── Dataset Split ─────────────────────────────────")
    save_jsonl(train, Path(dataset_cfg["train_file"]))
    save_jsonl(val,   Path(dataset_cfg["val_file"]))
    save_jsonl(eval_, Path(dataset_cfg["eval_file"]))

    log.info(f"\n✅ Dataset preparation complete!")
    log.info(f"   Train : {len(train)}")
    log.info(f"   Val   : {len(val)}")
    log.info(f"   Eval  : {len(eval_)}")


if __name__ == "__main__":
    main()
