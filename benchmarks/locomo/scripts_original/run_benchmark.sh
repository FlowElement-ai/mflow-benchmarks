#!/bin/bash
# M-flow LoCoMo Benchmark — One-click Reproduction
#
# Prerequisites:
#   1. M-flow installed (pip install -e . from M-Flow repo root)
#   2. Environment variables set in .env or exported:
#      - LLM_API_KEY (OpenAI API key)
#      - OPENAI_API_KEY (same key, for evaluation LLM)
#   3. Dataset: dataset/locomo10.json in repo root
#
# Usage:
#   cd benchmarks/locomo
#   bash run_benchmark.sh
#
# Total time: ~4 hours (ingest ~2.5h, search ~1h, eval ~30min)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
MFLOW_DIR="$(dirname "$(dirname "$BENCHMARK_DIR")")"
DATA_PATH="${MFLOW_DIR}/../dataset/locomo10.json"
RESULTS_DIR="${SCRIPT_DIR}/results"
VENV_PYTHON="${MFLOW_DIR}/.venv/bin/python"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "M-flow LoCoMo Benchmark"
echo "============================================================"
echo "M-flow dir:  $MFLOW_DIR"
echo "Data:        $DATA_PATH"
echo "Results:     $RESULTS_DIR"
echo "Python:      $VENV_PYTHON"
echo "============================================================"

# Check prerequisites
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Dataset not found at $DATA_PATH"
    exit 1
fi
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: M-flow venv not found. Run: python -m venv .venv && .venv/bin/pip install -e . (from M-Flow repo)"
    exit 1
fi

# Check no uvicorn running (Kuzu lock)
if pgrep -f "uvicorn.*m_flow" > /dev/null 2>&1; then
    echo "ERROR: M-flow API server is running. Stop it first (Kuzu file lock)."
    exit 1
fi

# ============================================================
# Step 1: Ingest (skip if data already exists)
# ============================================================
echo ""
echo "[Step 1/3] Ingesting LoCoMo data..."
echo "This step takes ~2.5 hours. Skippable if data is already ingested."
read -p "Skip ingestion? (y/N): " SKIP_INGEST
if [ "${SKIP_INGEST:-n}" != "y" ]; then
    cd "$BENCHMARK_DIR"
    PYTHONUNBUFFERED=1 "$VENV_PYTHON" -u run_ingest_batched.py \
        --no-prune --force \
        --data "$DATA_PATH" \
        2>&1 | tee "$RESULTS_DIR/ingest.log"
    echo "[Step 1] Ingestion complete."
else
    echo "[Step 1] Skipped."
fi

# ============================================================
# Step 2: Search + Answer
# ============================================================
echo ""
echo "[Step 2/3] Running search + answer (all 10 conversations)..."
cd "$BENCHMARK_DIR"
PYTHONUNBUFFERED=1 "$VENV_PYTHON" -u search_aligned.py \
    --data-path "$DATA_PATH" \
    --output-path "$RESULTS_DIR/mflow_search.json" \
    --top-k 10 \
    2>&1 | tee "$RESULTS_DIR/search.log"
echo "[Step 2] Search complete."

# ============================================================
# Step 3: Evaluate
# ============================================================
echo ""
echo "[Step 3/3] Evaluating results..."
cd "$BENCHMARK_DIR"
PYTHONUNBUFFERED=1 "$VENV_PYTHON" -u evaluate_aligned.py \
    --input-file "$RESULTS_DIR/mflow_search.json" \
    --output-file "$RESULTS_DIR/mflow_eval.json" \
    2>&1 | tee "$RESULTS_DIR/eval.log"
echo "[Step 3] Evaluation complete."

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
echo "Results: $RESULTS_DIR/mflow_eval.json"
echo ""
"$VENV_PYTHON" -c "
import json
with open('$RESULTS_DIR/mflow_eval.json') as f:
    data = json.load(f)
tp = sum(e.get('llm_score',0) for v in data.values() for e in v)
te = sum(len(v) for v in data.values())
bleu = sum(e.get('bleu_score',0) for v in data.values() for e in v) / te
f1 = sum(e.get('f1_score',0) for v in data.values() for e in v) / te
print(f'LLM-Judge: {tp}/{te} = {tp/te*100:.1f}%')
print(f'BLEU-1:    {bleu:.3f}')
print(f'F1:        {f1:.3f}')
print()
for k in sorted(data.keys(), key=int):
    items = data[k]
    p = sum(e.get('llm_score',0) for e in items)
    print(f'  Conv {k}: {p}/{len(items)} = {p/len(items)*100:.1f}%')
"
