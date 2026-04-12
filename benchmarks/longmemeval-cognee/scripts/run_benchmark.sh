#!/usr/bin/env bash
#
# Cognee LongMemEval Benchmark — End-to-End Runner
#
# Prerequisites:
#   1. Docker Desktop running with Cognee deployed (see docker/ directory)
#   2. Environment variables set (see .env.example)
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/run_benchmark.sh [--questions 100]

set -euo pipefail
cd "$(dirname "$0")/.."

QUESTIONS=${1:-100}
if [[ "${1:-}" == "--questions" ]]; then
    QUESTIONS=${2:-100}
fi

COGNEE_URL="${COGNEE_LOCAL_URL:-http://localhost:8001}"

echo "============================================================"
echo "  Cognee LongMemEval Benchmark"
echo "============================================================"
echo "  Cognee URL:  $COGNEE_URL"
echo "  Questions:   $QUESTIONS"
echo "  Answer:      gpt-5-mini"
echo "  Judge:       gpt-4o-mini"
echo "============================================================"
echo ""

# --- Pre-flight checks ---

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi

echo "[1/4] Checking Cognee health..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$COGNEE_URL/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "  ERROR: Cognee is not reachable at $COGNEE_URL (HTTP $HTTP_CODE)"
    echo "  Please start Cognee: cd docker && docker compose up -d"
    exit 1
fi
echo "  Cognee is healthy."

echo "[2/4] Installing dependencies..."
pip install -q -r requirements.txt
echo "  Done."

# --- Ingestion ---

echo ""
echo "[3/4] Ingesting data ($QUESTIONS questions)..."
python -u scripts/cognee_ingest.py \
    --max-questions "$QUESTIONS" \
    --chunk-size 2000 \
    --delay 2

# --- Evaluation ---

echo ""
echo "[4/4] Evaluating ($QUESTIONS questions)..."
python -u scripts/cognee_evaluate.py \
    --max-questions "$QUESTIONS" \
    --answer-model gpt-5-mini \
    --judge-model gpt-4o-mini \
    --top-k 10 \
    --delay 0.5

# --- Analysis ---

echo ""
echo "Analyzing results..."
python scripts/analyze_results.py results/cognee_eval_results.json --export

echo ""
echo "============================================================"
echo "  Benchmark complete! Results in results/"
echo "============================================================"
