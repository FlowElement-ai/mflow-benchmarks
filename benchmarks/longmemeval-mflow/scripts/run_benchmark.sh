#!/usr/bin/env bash
# M-flow LongMemEval Benchmark — orchestrator (runs on host, delegates to Docker)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER="${MFLOW_CONTAINER:-m_flow}"

: "${OPENAI_API_KEY:?Error: OPENAI_API_KEY is not set}"

echo "============================================================"
echo "  M-flow LongMemEval Benchmark"
echo "============================================================"
echo "  Container:    $CONTAINER"
echo "  Answer model: gpt-5-mini"
echo "  Judge model:  gpt-4o-mini"
echo "  Mode:         precise_mode=True"
echo "============================================================"

# Verify container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Error: Container '$CONTAINER' is not running."
    echo "Start M-flow first: cd <mflow-dir> && docker compose up -d"
    exit 1
fi

# Verify health
HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null || echo '{}')
if ! echo "$HEALTH" | grep -q "ready"; then
    echo "Error: M-flow API not ready. Health: $HEALTH"
    exit 1
fi
echo "M-flow API: ready"

# Copy scripts and data to container
echo ""
echo "[Setup] Copying scripts and data to container..."
docker exec "$CONTAINER" mkdir -p /opt/benchmark/scripts /opt/benchmark/data /opt/benchmark/results
docker cp "$BENCH_DIR/scripts/ingest.py" "$CONTAINER:/opt/benchmark/scripts/"
docker cp "$BENCH_DIR/scripts/evaluate.py" "$CONTAINER:/opt/benchmark/scripts/"
docker cp "$BENCH_DIR/scripts/collect_retrieval.py" "$CONTAINER:/opt/benchmark/scripts/"
docker cp "$BENCH_DIR/scripts/run_ingest.sh" "$CONTAINER:/opt/benchmark/scripts/"
docker cp "$BENCH_DIR/data/longmemeval_oracle.json" "$CONTAINER:/opt/benchmark/data/"
docker exec "$CONTAINER" chmod +x /opt/benchmark/scripts/run_ingest.sh

MAX_Q="${MAX_QUESTIONS:-100}"

# Step 1: Ingest
echo ""
echo "[Step 1/3] Ingesting $MAX_Q questions (precise_mode=True)..."
docker exec "$CONTAINER" bash /opt/benchmark/scripts/run_ingest.sh --max-questions "$MAX_Q" --start-from 0

# Step 2: Evaluate
echo ""
echo "[Step 2/3] Evaluating $MAX_Q questions..."
docker exec "$CONTAINER" bash -c "
export MFLOW_ROOT=/opt/m_flow VECTOR_DB_URL='' OPENAI_API_KEY='$OPENAI_API_KEY'
cd /opt/m_flow
python3 -u /opt/benchmark/scripts/evaluate.py \
  --max-questions $MAX_Q --start-from 0 \
  --answer-model gpt-5-mini --judge-model gpt-4o-mini \
  --output-dir /opt/benchmark/results
"

# Step 3: Collect retrieval
echo ""
echo "[Step 3/3] Collecting retrieval content..."
docker exec "$CONTAINER" bash -c "
export MFLOW_ROOT=/opt/m_flow VECTOR_DB_URL=''
cd /opt/m_flow
python3 -u /opt/benchmark/scripts/collect_retrieval.py \
  --max-questions $MAX_Q --start-from 0 \
  --output /opt/benchmark/results/retrieval.json
"

# Copy results back
echo ""
echo "[Done] Copying results to host..."
docker cp "$CONTAINER:/opt/benchmark/results/" "$BENCH_DIR/results_raw/"

echo ""
echo "============================================================"
echo "  Benchmark complete. Raw results in results_raw/"
echo "============================================================"
