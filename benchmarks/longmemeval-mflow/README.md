# M-flow LongMemEval Benchmark

Benchmark evaluation of [M-flow](https://github.com/mflow-ai/m_flow) (v0.3.2) with `precise_mode` on the [LongMemEval](https://arxiv.org/abs/2407.15167) Oracle dataset.

## Results (First 100 Questions)

| Metric | Value |
|--------|-------|
| **LLM-Judge Accuracy** | **89.0%** (89/100) |
| BLEU-1 | 0.3046 |
| F1 (token-level) | 0.4397 |

### By Question Type

| Type | Questions | Accuracy |
|------|-----------|----------|
| temporal-reasoning | 60 | **93%** |
| multi-session | 40 | **82%** |

## Configuration

| Parameter | Value |
|-----------|-------|
| M-flow version | 0.3.2 |
| Deployment | Docker (self-hosted) |
| `precise_mode` | True |
| `episode_routing` | False |
| Ingestion LLM | gpt-5-nano |
| Embedding | text-embedding-3-small (1536d) |
| Answer model | gpt-5-mini |
| Judge model | gpt-4o-mini |
| Graph DB | KuzuDB (per-dataset) |
| Vector DB | LanceDB (per-dataset) |

## Reproduction

### 1. Deploy M-flow

```bash
git clone https://github.com/mflow-ai/m_flow.git
cd m_flow
cp /path/to/benchmark/.env.benchmark .env
# Edit .env: set your API keys
docker compose build --no-cache
docker compose up -d
```

### 2. Apply Precise Mode Patches

```bash
bash scripts/apply_patches.sh /path/to/m_flow
# Restart to pick up changes (gunicorn --reload does this automatically)
```

### 3. Run Benchmark

```bash
export OPENAI_API_KEY="sk-..."
bash scripts/run_benchmark.sh
```

Or step by step:

```bash
# Ingest (runs inside Docker container)
docker exec m_flow bash scripts/run_ingest.sh --max-questions 100

# Evaluate
docker exec m_flow bash -c '
  export OPENAI_API_KEY="sk-..." MFLOW_ROOT=/opt/m_flow VECTOR_DB_URL=""
  cd /opt/m_flow
  python3 /opt/benchmark/scripts/evaluate.py \
    --max-questions 100 --answer-model gpt-5-mini --judge-model gpt-4o-mini
'

# Collect retrieval content
docker exec m_flow bash -c '
  export MFLOW_ROOT=/opt/m_flow VECTOR_DB_URL=""
  cd /opt/m_flow
  python3 /opt/benchmark/scripts/collect_retrieval.py --max-questions 100
'
```

## File Structure

```
├── scripts/
│   ├── ingest.py              # Ingestion (precise_mode=True)
│   ├── evaluate.py            # Evaluation (retrieve → answer → judge)
│   ├── collect_retrieval.py   # Retrieval content collection
│   ├── run_ingest.sh          # Docker container ingestion wrapper
│   ├── run_benchmark.sh       # One-shot orchestrator
│   └── apply_patches.sh       # Apply precise_mode to M-flow source
├── patches/                    # Precise mode source modifications (9 files)
│   └── README.md              # Modification summary
├── data/
│   └── longmemeval_oracle.json
├── results/
│   ├── eval_100_per_question.json   # Per-question: retrieval + answer + judge
│   ├── eval_100_summary.json        # Aggregate metrics
│   ├── eval_by_type.json            # By question type
│   ├── ingest_summary.json
│   └── config.json
├── docs/
│   └── precise_mode.md        # Design document + performance analysis
├── .env.benchmark              # M-flow configuration template
└── README.md
```

## Per-Question Result Fields

Each entry in `eval_100_per_question.json`:

| Field | Description |
|-------|-------------|
| `question_id` | Unique identifier |
| `question_type` | temporal-reasoning, multi-session, etc. |
| `question` / `gold_answer` | Question text and ground truth |
| `retrieved_memories` | Full retrieval text from EpisodicRetriever |
| `memories_count` | Number of memory segments retrieved |
| `generated_answer` | Model-generated answer |
| `bleu_score` / `f1_score` / `llm_score` | Evaluation metrics |
| `retrieval_ms` / `generation_ms` / `total_ms` | Latency |

## About Precise Mode

See [docs/precise_mode.md](docs/precise_mode.md) for the design, implementation details, and performance comparison with the default pipeline.

## Known Limitations

- `gpt-5-mini` does not support `temperature=0`; results may vary across runs
- Ingestion is slow (~20 min/question) due to multi-step LLM processing
- Current results cover 100/500 questions (temporal-reasoning + multi-session types)
- The `retrieved_memories` field was collected post-evaluation; `memories_count` may differ slightly from evaluation-time count due to different counting methods


## Comparisons

LongMemEval Oracle — First 100 Questions | Answer: gpt-5-mini | Judge: gpt-4o-mini | Top-K: 10

| System | Overall | Temporal (60) | Multi-session (40) |
|--------|:-------:|:-------------:|:------------------:|
| **M-flow** | **89%** | **93%** | **82%** |
| Supermemory Cloud | 74% | 78% | 68% |
| Mem0 | 71% | 77% | 63% |
| Zep Cloud | 61% | 82% | 30% |
| Cognee | 57% | 67% | 43% |
