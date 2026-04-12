#!/bin/bash
#
# Mem0 LongMemEval Benchmark Runner
# 对齐 run_benchmark.sh (MFlow) 的结构和参数
#
# 用法:
#   ./run_mem0_benchmark.sh                         # 默认运行 50 题
#   ./run_mem0_benchmark.sh --questions 100          # 运行 100 题
#   ./run_mem0_benchmark.sh --skip-ingest            # 跳过入库，仅评估
#   ./run_mem0_benchmark.sh --help                   # 帮助
#

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

# 默认参数 — 与 MFlow benchmark 完全一致
MAX_QUESTIONS=50
SKIP_INGEST=false
SKIP_EVAL=false
ANSWER_MODEL="gpt-5-mini"
JUDGE_MODEL="gpt-4o-mini"
SESSION_DELAY=2.0
API_DELAY=0.5
TOP_K=10
CLEAN_BEFORE_INGEST=false
# mem0 异步处理等待时间（入库完成后，等待 mem0 后台处理记忆）
# 注意: mem0 add() 始终异步 (async_mode=False 已废弃)，单条记忆处理约需 20-30 秒
PROCESSING_WAIT=120

# API Keys — 可通过命令行或环境变量设置
MEM0_KEY="${MEM0_API_KEY:-}"
OPENAI_KEY="${OPENAI_API_KEY:-}"

# ============================================================================
# Parse arguments
# ============================================================================

print_help() {
    cat << EOF
Mem0 LongMemEval Benchmark Runner
对齐 MFlow benchmark 的完整评估流程

Usage: $0 [OPTIONS]

Options:
  --questions N         评估问题数量 (默认: 50)
  --skip-ingest         跳过入库步骤
  --skip-eval           跳过评估步骤 (仅入库)
  --answer-model M      答题模型 (默认: gpt-5-mini)
  --judge-model M       评判模型 (默认: gpt-4o-mini)
  --top-k K             检索记忆数 (默认: 10)
  --session-delay S     入库会话间延迟秒数 (默认: 2.0)
  --api-delay S         评估 API 调用间延迟秒数 (默认: 0.5)
  --processing-wait S   入库后等待 mem0 处理的秒数 (默认: 120)
  --clean               入库前清除旧记忆
  --mem0-key KEY        Mem0 API Key (或设 MEM0_API_KEY 环境变量)
  --openai-key KEY      OpenAI API Key (或设 OPENAI_API_KEY 环境变量)
  --help                显示帮助

Examples:
  $0                                    # 完整测试 50 题
  $0 --questions 5                      # 快速冒烟测试 5 题
  $0 --skip-ingest                      # 仅评估 (已入库)
  $0 --clean --questions 50             # 清除旧数据后重新测试

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --questions)
            MAX_QUESTIONS="$2"
            shift 2
            ;;
        --skip-ingest)
            SKIP_INGEST=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --answer-model)
            ANSWER_MODEL="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --session-delay)
            SESSION_DELAY="$2"
            shift 2
            ;;
        --api-delay)
            API_DELAY="$2"
            shift 2
            ;;
        --processing-wait)
            PROCESSING_WAIT="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BEFORE_INGEST=true
            shift
            ;;
        --mem0-key)
            MEM0_KEY="$2"
            shift 2
            ;;
        --openai-key)
            OPENAI_KEY="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Validate API Keys
# ============================================================================

if [[ -z "$MEM0_KEY" ]]; then
    echo "ERROR: Mem0 API Key 未设置"
    echo "  方式1: $0 --mem0-key m0-xxx"
    echo "  方式2: export MEM0_API_KEY=m0-xxx"
    exit 1
fi

if [[ -z "$OPENAI_KEY" ]]; then
    echo "ERROR: OpenAI API Key 未设置"
    echo "  方式1: $0 --openai-key sk-xxx"
    echo "  方式2: export OPENAI_API_KEY=sk-xxx"
    exit 1
fi

export MEM0_API_KEY="$MEM0_KEY"
export OPENAI_API_KEY="$OPENAI_KEY"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
INGEST_LOG="${LOG_DIR}/mem0_ingest_${TIMESTAMP}.log"
EVAL_LOG="${LOG_DIR}/mem0_eval_${TIMESTAMP}.log"

# ============================================================================
# Banner
# ============================================================================

echo "============================================================"
echo "Mem0 LongMemEval Benchmark"
echo "============================================================"
echo "Timestamp:         ${TIMESTAMP}"
echo "Questions:         ${MAX_QUESTIONS}"
echo "Skip Ingest:       ${SKIP_INGEST}"
echo "Skip Eval:         ${SKIP_EVAL}"
echo "Clean Before:      ${CLEAN_BEFORE_INGEST}"
echo "Answer Model:      ${ANSWER_MODEL}"
echo "Judge Model:       ${JUDGE_MODEL}"
echo "Top-K:             ${TOP_K}"
echo "Session Delay:     ${SESSION_DELAY}s"
echo "Processing Wait:   ${PROCESSING_WAIT}s"
echo "API Delay:         ${API_DELAY}s"
echo "Results Dir:       ${RESULTS_DIR}"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Check prerequisites
# ============================================================================

echo "[1/6] Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  Python version: ${PYTHON_VERSION}"

python3 -c "import mem0; print(f'  mem0ai: OK (version: {mem0.__version__})')" 2>/dev/null || {
    python3 -c "import mem0; print('  mem0ai: OK')" 2>/dev/null || {
        echo "ERROR: mem0ai package not installed"
        echo "  pip install mem0ai"
        exit 1
    }
}

python3 -c "import openai" 2>/dev/null || {
    echo "ERROR: openai package not installed"
    echo "  pip install openai"
    exit 1
}
echo "  openai: OK"

python3 -c "import nltk" 2>/dev/null || {
    echo "ERROR: nltk package not installed"
    echo "  pip install nltk"
    exit 1
}
echo "  nltk: OK"

# 检查数据文件
DATA_FILE="${PROJECT_ROOT}/data/longmemeval_oracle.json"
if [[ -f "$DATA_FILE" ]]; then
    Q_COUNT=$(python3 -c "import json; print(len(json.load(open('${DATA_FILE}'))))")
    echo "  数据文件: OK (${Q_COUNT} questions)"
else
    echo "  WARNING: 数据文件未找到: ${DATA_FILE}"
    echo "  脚本会尝试其他路径..."
fi

# 验证 mem0 API Key 连接
echo "  验证 Mem0 API 连接..."
python3 -c "
from mem0 import MemoryClient
try:
    c = MemoryClient(api_key='${MEM0_KEY}')
    c.search('test', filters={'user_id': '__test__'})
    print('  Mem0 API: OK')
except Exception as e:
    print(f'  Mem0 API: WARNING - {e}')
    print('  (将继续运行)')
" 2>/dev/null || echo "  Mem0 API: 跳过验证"

echo "  Prerequisites OK"
echo ""

# ============================================================================
# Step 2: Data Ingestion
# ============================================================================

if [[ "$SKIP_INGEST" == "false" ]]; then
    echo "[2/6] Running Mem0 data ingestion..."
    echo "  Log: ${INGEST_LOG}"

    START_TIME=$(date +%s)

    INGEST_ARGS="--max-questions $MAX_QUESTIONS --session-delay $SESSION_DELAY --api-key $MEM0_KEY"
    if [[ "$CLEAN_BEFORE_INGEST" == "true" ]]; then
        INGEST_ARGS="$INGEST_ARGS --clean"
    fi

    python3 "${SCRIPT_DIR}/mem0_ingest.py" $INGEST_ARGS \
        2>&1 | tee "$INGEST_LOG"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "  Ingestion completed in ${ELAPSED} seconds"
    echo ""
else
    echo "[2/6] Skipping data ingestion (--skip-ingest)"
    echo ""
fi

# ============================================================================
# Step 3: Wait for mem0 async processing
# ============================================================================

if [[ "$SKIP_INGEST" == "false" && "$SKIP_EVAL" == "false" && "$PROCESSING_WAIT" -gt 0 ]]; then
    echo "[3/6] Waiting ${PROCESSING_WAIT}s for mem0 to finish processing memories..."
    echo "  (mem0 add() 始终异步处理, async_mode=False 已废弃)"
    echo "  单条记忆处理约需 20-30 秒，等待确保全部就绪"
    echo "  开始时间: $(date '+%H:%M:%S')"

    WAITED=0
    while [[ $WAITED -lt $PROCESSING_WAIT ]]; do
        REMAINING=$((PROCESSING_WAIT - WAITED))
        printf "  等待中... %3ds 剩余\r" "$REMAINING"
        sleep 5
        WAITED=$((WAITED + 5))
    done
    echo "  等待完成: $(date '+%H:%M:%S')                    "

    # 验证记忆是否可用（抽查第一个问题）
    echo "  验证记忆可用性..."
    python3 -c "
import json
from mem0 import MemoryClient

client = MemoryClient(api_key='${MEM0_KEY}')
data_path = '${PROJECT_ROOT}/data/longmemeval_oracle.json'

try:
    with open(data_path) as f:
        questions = json.load(f)
    q = questions[0]
    uid = f\"lme_{q['question_id']}\"
    r = client.search('test query', filters={'user_id': uid}, top_k=3)
    memories = r.get('results', r.get('memories', []))
    if memories:
        print(f'  验证通过: user_id={uid} 检索到 {len(memories)} 条记忆')
    else:
        print(f'  [WARN] user_id={uid} 未检索到记忆，可能需要更长等待时间')
        print(f'  可用 --processing-wait N 增大等待秒数')
except Exception as e:
    print(f'  验证跳过: {e}')
" 2>/dev/null || echo "  验证跳过"
    echo ""
else
    echo "[3/6] Skipping processing wait"
    echo ""
fi

# ============================================================================
# Step 4: QA Evaluation
# ============================================================================

if [[ "$SKIP_EVAL" == "false" ]]; then
    echo "[4/6] Running Mem0 QA evaluation..."
    echo "  Log: ${EVAL_LOG}"

    START_TIME=$(date +%s)

    python3 "${SCRIPT_DIR}/mem0_qa_eval.py" \
        --max-questions "$MAX_QUESTIONS" \
        --answer-model "$ANSWER_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --top-k "$TOP_K" \
        --api-delay "$API_DELAY" \
        --mem0-api-key "$MEM0_KEY" \
        --openai-api-key "$OPENAI_KEY" \
        --output-dir "$RESULTS_DIR" \
        2>&1 | tee "$EVAL_LOG"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "  Evaluation completed in ${ELAPSED} seconds"
    echo ""
else
    echo "[4/6] Skipping evaluation (--skip-eval)"
    echo ""
fi

# ============================================================================
# Step 5: Check Results
# ============================================================================

echo "[5/6] Checking results..."

RESULTS_FILE="${RESULTS_DIR}/mem0_eval_results.json"

if [[ -f "$RESULTS_FILE" ]]; then
    echo "  Results file: ${RESULTS_FILE}"

    python3 -c "
import json, sys
try:
    with open('${RESULTS_FILE}') as f:
        data = json.load(f)
    s = data['summary']
    print(f\"  Total Questions: {s['total_questions']}\")
    print(f\"  LLM-Judge Accuracy: {s['llm_accuracy']*100:.1f}%\")
    print(f\"  BLEU-1: {s['avg_bleu']:.4f}\")
    print(f\"  F1: {s['avg_f1']:.4f}\")
    print(f\"  Avg Retrieval Latency: {s['avg_retrieval_ms']:.2f}ms\")
    print(f\"  Total Time: {s['total_time_seconds']:.2f}s\")
    if 'per_type_accuracy' in s:
        print(f\"  ---\")
        for qt, st in sorted(s['per_type_accuracy'].items()):
            print(f\"  {qt}: {st['correct']}/{st['total']} ({st['accuracy']*100:.1f}%)\")
except Exception as e:
    print(f'  读取结果失败: {e}', file=sys.stderr)
"
else
    echo "  WARNING: Results file not found at ${RESULTS_FILE}"
fi

echo ""

# ============================================================================
# Step 6: Compare with MFlow (if available)
# ============================================================================

echo "[6/6] Benchmark Complete"
echo "============================================================"

MFLOW_RESULTS="${RESULTS_DIR}/mflow_eval_results.json"

if [[ -f "$MFLOW_RESULTS" && -f "$RESULTS_FILE" ]]; then
    echo ""
    echo "MFlow 结果已存在，运行对比分析..."
    python3 "${SCRIPT_DIR}/analyze_results.py" \
        --compare "$MFLOW_RESULTS" "$RESULTS_FILE" \
        2>/dev/null || echo "  对比分析跳过 (analyze_results.py 不可用)"
fi

echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo "Files:"
ls -la "${RESULTS_DIR}"/mem0_*.json 2>/dev/null || echo "  No result files found"
echo ""
echo "To analyze results:"
echo "  python3 ${SCRIPT_DIR}/analyze_results.py ${RESULTS_FILE}"
echo ""
echo "To compare with MFlow:"
echo "  python3 ${SCRIPT_DIR}/analyze_results.py --compare ${MFLOW_RESULTS} ${RESULTS_FILE}"
echo ""
echo "============================================================"
