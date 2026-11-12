"""
collect_arxiv.py
─────────────────
Collects research paper abstracts from arXiv using the arxiv Python client.
Papers are fetched from space-domain categories and saved as a JSONL file
under data/raw/.

What you will learn here:
  • Using an API client (arxiv) to query a research database
  • Saving structured data to JSONL (JSON Lines) format
  • Basic retry / error handling for network calls
  • Progress tracking with tqdm
"""

import arxiv
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Generator

import yaml
from tqdm import tqdm

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arxiv_query(category: str, date_from: str, date_to: str) -> str:
    """
    Build an arXiv search query string.

    arXiv supports date filtering via submittedDate field:
      submittedDate:[YYYYMMDD TO YYYYMMDD]
    """
    # Convert "2018-01-01" → "20180101"
    d_from = date_from.replace("-", "")
    d_to   = date_to.replace("-", "")
    return f"cat:{category} AND submittedDate:[{d_from} TO {d_to}]"


def fetch_papers(
    query: str,
    max_results: int,
    delay_seconds: float = 1.0,
) -> Generator[dict, None, None]:
    """
    Fetch papers from arXiv and yield them as dictionaries.

    Each yielded dict contains:
      - arxiv_id    : unique paper identifier
      - title       : paper title
      - abstract    : full abstract text
      - authors     : list of author names
      - categories  : list of arXiv category tags
      - published   : publication date (ISO string)
      - url         : paper URL
    """
    client = arxiv.Client(
        page_size=100,       # fetch 100 papers per API page
        delay_seconds=delay_seconds,
        num_retries=5,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    for result in client.results(search):
        yield {
            "arxiv_id":   result.entry_id.split("/")[-1],
            "title":      result.title.strip().replace("\n", " "),
            "abstract":   result.summary.strip().replace("\n", " "),
            "authors":    [str(a) for a in result.authors],
            "categories": result.categories,
            "published":  result.published.isoformat() if result.published else None,
            "url":        result.pdf_url,
        }


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Append records to a JSONL file (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config()
    arxiv_cfg   = cfg["arxiv"]
    categories  = arxiv_cfg["categories"]
    max_results = arxiv_cfg["max_results_per_category"]
    date_from   = arxiv_cfg["date_from"]
    date_to     = arxiv_cfg["date_to"]
    output_path = Path(arxiv_cfg["output_file"])

    # Remove existing file so we start fresh each run
    if output_path.exists():
        log.warning(f"Removing existing file: {output_path}")
        output_path.unlink()

    total_collected = 0

    for category in categories:
        log.info(f"Fetching category: {category}  (max {max_results} papers)")
        query   = build_arxiv_query(category, date_from, date_to)
        batch   = []

        try:
            for paper in tqdm(
                fetch_papers(query, max_results),
                total=max_results,
                desc=category,
                unit="paper",
            ):
                paper["source_category"] = category
                batch.append(paper)

        except Exception as exc:
            log.error(f"Error fetching {category}: {exc}")

        save_jsonl(batch, output_path)
        log.info(f"  Saved {len(batch)} papers from {category}")
        total_collected += len(batch)

        # Be polite to the arXiv API between categories
        time.sleep(3)

    log.info(f"\n✅ Done! Total papers collected: {total_collected}")
    log.info(f"   Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
