#!/usr/bin/env bash
# Supermemory LongMemEval Benchmark — one-shot runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Validate environment ──
: "${SUPERMEMORY_API_KEY:?Error: SUPERMEMORY_API_KEY is not set}"
: "${OPENAI_API_KEY:?Error: OPENAI_API_KEY is not set}"

echo "============================================================"
echo "  Supermemory LongMemEval Benchmark"
echo "============================================================"
echo "  Dataset:      LongMemEval Oracle (500 questions)"
echo "  Answer model: gpt-5-mini"
echo "  Judge model:  gpt-4o-mini"
echo "============================================================"

# ── Step 1: Ingestion ──
echo ""
echo "[Step 1/2] Ingesting 500 questions into Supermemory..."
python3 scripts/ingest.py --max-questions 500 --start-from 0

# ── Step 2: Evaluation ──
echo ""
echo "[Step 2/2] Evaluating 500 questions..."
python3 scripts/evaluate.py \
  --max-questions 500 --start-from 0 \
  --answer-model gpt-5-mini --judge-model gpt-4o-mini

echo ""
echo "============================================================"
echo "  Benchmark complete. Results in results/"
echo "============================================================"
