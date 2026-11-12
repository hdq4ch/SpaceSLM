"""
generate.py
────────────
Local CPU inference using the GGUF quantized SpaceSLM model via llama-cpp-python.

What you will learn here:
  • Loading GGUF models with llama-cpp-python
  • Building a chat-style prompt for instruction-tuned models
  • Controlling generation (temperature, top-p, repetition penalty)
  • Measuring inference latency and tokens/sec
"""

import argparse
import logging
import time
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_prompt(instruction: str, input_text: str = "") -> str:
    """
    Format a prompt in the Alpaca / Qwen instruction style.
    The model was fine-tuned on this exact template, so we must match it.
    """
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


def run_inference(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    n_ctx: int = 512,
) -> tuple[str, float, float]:
    """
    Run inference and return (generated_text, latency_seconds, tokens_per_sec).

    Parameters
    ----------
    model_path     : path to the GGUF model file
    prompt         : the full formatted prompt string
    max_new_tokens : maximum number of tokens to generate
    temperature    : sampling temperature (lower = more deterministic)
    top_p          : nucleus sampling threshold
    repeat_penalty : penalise repeated tokens (> 1.0 reduces repetition)
    n_ctx          : context window size in tokens
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed.\n"
            "Install it with:  pip install llama-cpp-python\n"
            "For CPU-only:     pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"GGUF model not found: {model_path}\n"
            "Run the GGUF conversion step first."
        )

    log.info(f"Loading GGUF model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=4,     # number of CPU threads – adjust to your machine
        verbose=False,
    )

    log.info("Generating response...")
    start = time.perf_counter()

    output = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stop=["### Instruction:", "### Input:"],  # stop tokens
        echo=False,
    )

    elapsed = time.perf_counter() - start
    generated = output["choices"][0]["text"].strip()
    n_tokens  = output["usage"]["completion_tokens"]
    tps       = n_tokens / elapsed if elapsed > 0 else 0.0

    return generated, elapsed, tps


def interactive_session(cfg: dict) -> None:
    """
    Simple REPL for chatting with the local SpaceSLM model.
    Type 'quit' or 'exit' to stop.
    """
    inf_cfg    = cfg["inference"]
    model_path = inf_cfg["gguf_model_path"]

    print("\n" + "═" * 60)
    print("  🚀  SpaceSLM-mini – Local Inference")
    print("  Model:", model_path)
    print("  Type 'quit' to exit")
    print("═" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if not question:
            continue

        prompt = build_prompt(question)

        try:
            answer, latency, tps = run_inference(
                model_path=model_path,
                prompt=prompt,
                max_new_tokens=inf_cfg["max_new_tokens"],
                temperature=inf_cfg["temperature"],
                top_p=inf_cfg["top_p"],
                repeat_penalty=inf_cfg["repeat_penalty"],
                n_ctx=inf_cfg["n_ctx"],
            )
            print(f"\n🤖 SpaceSLM: {answer}")
            print(f"   ⏱  {latency:.2f}s │ {tps:.1f} tok/s\n")

        except Exception as e:
            log.error(f"Inference error: {e}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaceSLM local inference")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question to answer (non-interactive mode)")
    args = parser.parse_args()

    cfg = load_config()

    if args.question:
        # Non-interactive single-question mode
        prompt = build_prompt(args.question)
        answer, latency, tps = run_inference(
            model_path=cfg["inference"]["gguf_model_path"],
            prompt=prompt,
            max_new_tokens=cfg["inference"]["max_new_tokens"],
        )
        print(f"\n🤖 Answer: {answer}")
        print(f"⏱  Latency: {latency:.2f}s  │  {tps:.1f} tok/s")
    else:
        interactive_session(cfg)
