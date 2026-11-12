#!/usr/bin/env bash
# convert_gguf.sh
# ─────────────────
# Converts a merged HuggingFace FP16 model to GGUF format using llama.cpp,
# then quantizes to Q4_K_M for efficient CPU inference.
#
# Prerequisites:
#   1. llama.cpp compiled on your system (or use the Python wheels)
#   2. Python environment with requirements installed
#   3. Merged FP16 model at models/merged_fp16/
#
# What you will learn:
#   • The difference between FP16, GGUF, and Q4_K_M formats
#   • How llama.cpp's convert_hf_to_gguf.py works
#   • How quantization reduces model size with minimal quality loss

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
MERGED_MODEL_DIR="models/merged_fp16"
GGUF_DIR="models/gguf"
MODEL_NAME="space-slm"

# llama.cpp paths – adjust these to where you cloned / compiled llama.cpp
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/opt/llama.cpp}"
CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

# ── Checks ─────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════"
echo "  SpaceSLM-mini → GGUF Conversion Pipeline"
echo "════════════════════════════════════════════════"

if [ ! -d "${MERGED_MODEL_DIR}" ]; then
    echo "❌ Merged model directory not found: ${MERGED_MODEL_DIR}"
    echo "   Run merge_adapter.py first."
    exit 1
fi

if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "❌ llama.cpp convert script not found: ${CONVERT_SCRIPT}"
    echo ""
    echo "   Setup steps:"
    echo "   1. git clone https://github.com/ggerganov/llama.cpp"
    echo "   2. cd llama.cpp && cmake -B build && cmake --build build --config Release"
    echo "   3. pip install -r llama.cpp/requirements.txt"
    echo "   Or set LLAMA_CPP_DIR environment variable."
    exit 1
fi

mkdir -p "${GGUF_DIR}"

# ── Step 1: Convert HF → GGUF (FP16) ─────────────────────────────────────────
echo ""
echo "Step 1/2 · Converting HuggingFace model → GGUF (FP16)..."
FP16_GGUF="${GGUF_DIR}/${MODEL_NAME}-f16.gguf"

python "${CONVERT_SCRIPT}" \
    "${MERGED_MODEL_DIR}" \
    --outfile "${FP16_GGUF}" \
    --outtype f16

echo "✅ FP16 GGUF saved: ${FP16_GGUF}"

# ── Step 2: Quantize FP16 → Q4_K_M ───────────────────────────────────────────
echo ""
echo "Step 2/2 · Quantizing FP16 GGUF → Q4_K_M..."
echo ""
echo "   Q4_K_M: 4-bit quantization with medium quality"
echo "   • Weights stored as 4-bit integers"
echo "   • K_M = 'k-quant medium' – best quality/size tradeoff"
echo "   • Typical size reduction: 75% vs FP32"
echo ""

Q4_GGUF="${GGUF_DIR}/${MODEL_NAME}-q4_k_m.gguf"

"${QUANTIZE_BIN}" \
    "${FP16_GGUF}" \
    "${Q4_GGUF}" \
    Q4_K_M

echo ""
echo "✅ Quantized model saved: ${Q4_GGUF}"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Conversion complete!"
echo ""
echo "  FP16 GGUF  : ${FP16_GGUF}"
echo "  Q4_K_M GGUF: ${Q4_GGUF}"
echo ""
echo "  File sizes:"
du -sh "${FP16_GGUF}" 2>/dev/null || echo "  (size unavailable)"
du -sh "${Q4_GGUF}"   2>/dev/null || echo "  (size unavailable)"
echo ""
echo "  Run inference:"
echo "  python src/inference/generate.py"
echo "════════════════════════════════════════════════"
