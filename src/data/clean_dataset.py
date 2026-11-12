"""
clean_dataset.py
─────────────────
Reads raw arXiv JSONL data, applies text cleaning rules, removes duplicates,
filters short / low-quality abstracts, and writes the cleaned output.

What you will learn here:
  • Text normalisation (unicode, whitespace, special chars)
  • Deduplication using hashing
  • Quality filtering with length and language heuristics
  • Reading / writing JSONL with the jsonlines library
"""

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

import yaml
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Convert unicode characters to their closest ASCII equivalent."""
    return unicodedata.normalize("NFKC", text)


def remove_latex(text: str) -> str:
    """
    Remove common LaTeX math and command patterns.
    arXiv abstracts often contain inline math like $E=mc^2$ or \textit{...}
    """
    # Remove display math: $$ ... $$
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    # Remove inline math: $ ... $
    text = re.sub(r"\$[^$]+?\$", "", text)
    # Remove LaTeX commands: \command{...} or \command
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove leftover braces
    text = re.sub(r"[{}]", "", text)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_special_chars(text: str) -> str:
    """Remove non-printable / control characters."""
    return "".join(ch for ch in text if ch.isprintable())


def clean_text(text: str) -> str:
    """Apply the full cleaning pipeline to a string."""
    text = normalize_unicode(text)
    text = remove_latex(text)
    text = remove_special_chars(text)
    text = clean_whitespace(text)
    return text


# ── Quality Filters ────────────────────────────────────────────────────────────

MIN_ABSTRACT_WORDS = 30    # skip abstracts shorter than this
MAX_ABSTRACT_WORDS = 600   # skip abstracts longer than this (likely parsing error)

SPACE_KEYWORDS = [
    "satellite", "spacecraft", "orbit", "space", "lunar", "mars",
    "asteroid", "telescope", "rocket", "launch", "mission", "payload",
    "propulsion", "solar", "planetary", "astronomical", "astrophysics",
    "cosmology", "exoplanet", "nasa", "esa", "cubesat", "debris",
    "reentry", "thruster", "attitude control", "navigation",
]

def is_space_related(text: str) -> bool:
    """
    Heuristic: abstract must contain at least one space-domain keyword.
    Helps filter out papers that ended up in a category by mistake.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in SPACE_KEYWORDS)


def passes_quality_filter(record: dict) -> tuple[bool, str]:
    """
    Return (passes, reason) for a paper record.
    'reason' explains why a record was rejected.
    """
    abstract = record.get("abstract", "").strip()

    if not abstract:
        return False, "empty abstract"

    words = abstract.split()
    if len(words) < MIN_ABSTRACT_WORDS:
        return False, f"too short ({len(words)} words)"
    if len(words) > MAX_ABSTRACT_WORDS:
        return False, f"too long ({len(words)} words)"
    if not is_space_related(abstract):
        return False, "not space-related"

    return True, "ok"


# ── Deduplication ──────────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """SHA-256 hash of lowercased, whitespace-collapsed text."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = {}
    try:
        with open("configs/training_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("Config not found, using default paths.")

    input_path  = Path(cfg.get("arxiv", {}).get("output_file", "data/raw/arxiv_papers.jsonl"))
    output_path = Path("data/processed/cleaned_papers.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        log.error("Run collect_arxiv.py first.")
        return

    seen_hashes: set[str] = set()
    stats = {"total": 0, "passed": 0, "duplicate": 0, "rejected": 0}

    with open(output_path, "w", encoding="utf-8") as out_f:
        with open(input_path, "r", encoding="utf-8") as in_f:
            lines = in_f.readlines()

        for line in tqdm(lines, desc="Cleaning", unit="paper"):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["total"] += 1

            # Clean text fields
            record["title"]    = clean_text(record.get("title", ""))
            record["abstract"] = clean_text(record.get("abstract", ""))

            # Quality filter
            passes, reason = passes_quality_filter(record)
            if not passes:
                stats["rejected"] += 1
                log.debug(f"Rejected [{reason}]: {record.get('arxiv_id', '?')}")
                continue

            # Deduplication
            h = content_hash(record["abstract"])
            if h in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(h)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["passed"] += 1

    log.info("\n── Cleaning Summary ──────────────────────────────")
    log.info(f"  Total processed : {stats['total']}")
    log.info(f"  Passed          : {stats['passed']}")
    log.info(f"  Rejected        : {stats['rejected']}")
    log.info(f"  Duplicates      : {stats['duplicate']}")
    log.info(f"  Output          : {output_path.resolve()}")


if __name__ == "__main__":
    main()
