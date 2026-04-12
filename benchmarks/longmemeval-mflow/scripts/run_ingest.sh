#!/bin/bash
# M-flow 入库启动脚本 — 在 Docker 容器内运行
# 修复 .env 中 VECTOR_DB_URL 内联注释问题
#
# 用法 (从宿主机):
#   docker exec -it m_flow bash /opt/benchmark/scripts/run_mflow_ingest.sh --max-questions 300
#
# 支持所有 mflow_ingest.py 的参数:
#   --max-questions N   最大处理问题数
#   --start-from N      从第 N 个问题开始 (断点续传)

set -e

export MFLOW_ROOT="/opt/m_flow"
export VECTOR_DB_URL=""

cd "$MFLOW_ROOT"

echo "============================================================"
echo "  M-flow 入库 (容器内运行)"
echo "============================================================"
echo "  MFLOW_ROOT:    $MFLOW_ROOT"
echo "  VECTOR_DB_URL: (empty, using local lancedb)"
echo "  参数:          $@"
echo "============================================================"

exec python3 /opt/benchmark/scripts/mflow_ingest.py "$@"
