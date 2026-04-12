# Cognee LongMemEval Oracle Benchmark

Benchmark evaluation of [Cognee](https://github.com/topoteretes/cognee) on the **LongMemEval Oracle** dataset (ICLR 2025).

Cognee is deployed locally via Docker and evaluated using the same prompts, metrics, and methodology as the mflow/mem0/zep benchmarks for direct comparability.

## Results (First 100 Questions)

| Metric | Score |
|--------|-------|
| **LLM-Judge Accuracy** | **57.0%** (57/100) |
| BLEU-1 | 0.2394 |
| F1 | 0.3292 |
| Answer Model | gpt-5-mini |
| Judge Model | gpt-4o-mini |

### By Question Type

| Type | Correct | Total | Accuracy |
|------|---------|-------|----------|
| temporal-reasoning | 40 | 60 | 66.7% |
| multi-session | 17 | 40 | 42.5% |

## Prerequisites

- **Docker Desktop** (running)
- **Python 3.10+**
- **OpenAI API Key** with access to `gpt-5-mini` and `gpt-4o-mini`

## Quick Start

### 1. Deploy Cognee Locally

```bash
cd docker
cp .env.example .env
# Edit .env — set LLM_API_KEY to your OpenAI key
docker compose up -d
# Wait for healthy status:
docker ps  # Should show "healthy"
```

### 2. Register a User

Cognee requires authentication. Register once:

```bash
COGNEE_URL=http://localhost:8001

# Register
curl -X POST "$COGNEE_URL/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "benchmark@gmail.com", "password": "Benchmark2026!"}'

# Verify login
curl -X POST "$COGNEE_URL/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=benchmark@gmail.com&password=Benchmark2026!"
```

### 3. Run the Benchmark

```bash
export OPENAI_API_KEY=sk-your-key
bash scripts/run_benchmark.sh --questions 100
```

Or run steps individually:

```bash
# Ingest
python scripts/cognee_ingest.py --max-questions 100

# Evaluate
python scripts/cognee_evaluate.py --max-questions 100

# Analyze
python scripts/analyze_results.py results/cognee_eval_results.json --export
```

## Reproduce Our Results

The `results/` directory contains our complete evaluation data. To verify:

```bash
python scripts/analyze_results.py results/cognee_eval_results_100.json
```

To reproduce from scratch, follow the Quick Start steps above. Note that gpt-5-mini is a reasoning model with non-deterministic output, so exact scores may vary slightly across runs.

## Configuration

### Cognee Deployment

| Parameter | Value |
|-----------|-------|
| Embedding model | `text-embedding-3-small` (1536d) |
| Chunk size | 2000 characters (sentence-aware) |
| Graph DB | KuzuDB |
| Vector DB | LanceDB |
| DB provider | SQLite |

### Evaluation

| Parameter | Value |
|-----------|-------|
| Answer model | `gpt-5-mini` |
| Judge model | `gpt-4o-mini` |
| Top-K retrieval | 10 |
| Dataset isolation | One dataset per question (`lme_{question_id}`) |

### gpt-5-mini Compatibility

gpt-5-mini is a reasoning model with API restrictions:
- `temperature` parameter is **not supported** (must be omitted)
- `max_tokens` is **not supported**; use `max_completion_tokens` instead
- `max_completion_tokens` includes internal reasoning tokens — set to ≥2048

The evaluation script handles these automatically.

## Methodology

### Alignment with mflow/mem0/zep

This benchmark uses **identical prompts and metrics** as the mflow, mem0, and zep evaluations:

- **Answer prompt**: Instructs the model to give a concise answer (<6 words) based on retrieved memories
- **Judge prompt**: Generous labeling — marks CORRECT if the answer captures the same meaning
- **Metrics**: LLM-Judge Accuracy (primary), BLEU-1, F1

### Pipeline

1. **Ingestion**: For each question, all conversation sessions are formatted with timestamps, chunked at ~2000 characters respecting sentence boundaries, uploaded to Cognee, and processed with `cognify` (builds knowledge graph + embeddings).

2. **Evaluation**: For each question, memories are retrieved via Cognee's search API, an answer is generated with gpt-5-mini, and correctness is judged by gpt-4o-mini.

## File Structure

```
cognee_longmemeval_benchmark/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml                # Benchmark configuration
├── .env.example               # Environment variable template
├── .gitignore
├── data/
│   ├── longmemeval_oracle.json       # 500 questions (14.7MB)
│   └── README.md
├── scripts/
│   ├── cognee_ingest.py              # Data ingestion
│   ├── cognee_evaluate.py            # QA evaluation
│   ├── analyze_results.py            # Results analysis
│   └── run_benchmark.sh              # End-to-end runner
├── results/
│   ├── cognee_eval_results_100.json  # First 100 Qs (summary + per-Q)
│   ├── cognee_eval_first100_detailed.json  # Detailed with full retrieval
│   ├── eval_summary.json             # Compact summary
│   ├── eval_by_type.json             # Per-type breakdown
│   └── cognee_ingest_map.json        # question_id -> dataset mapping
└── docker/
    ├── docker-compose.yml            # Cognee Docker deployment
    └── .env.example                  # Docker env template
```

## Per-Question Result Fields

Each entry in `cognee_eval_results_100.json` contains:

| Field | Description |
|-------|-------------|
| `question_id` | Unique question identifier |
| `question` | Full question text |
| `question_type` | `temporal-reasoning` or `multi-session` |
| `gold_answer` | Ground truth answer |
| `generated_answer` | Model-generated answer |
| `memories_retrieved` | Full text of retrieved memories |
| `dataset_name` | Cognee dataset name |
| `memories_count` | Number of memory segments |
| `bleu_score` | BLEU-1 score |
| `f1_score` | Token-level F1 |
| `llm_score` | LLM judge result (1=correct, 0=wrong) |
| `retrieval_ms` | Retrieval latency (ms) |
| `generation_ms` | Answer generation latency (ms) |
| `total_ms` | Total latency (ms) |

## Comparisons

LongMemEval Oracle — First 100 Questions | Answer: gpt-5-mini | Judge: gpt-4o-mini | Top-K: 10

| System | Overall | Temporal (60) | Multi-session (40) |
|--------|:-------:|:-------------:|:------------------:|
| **M-flow** | **89%** | **93%** | **82%** |
| Supermemory Cloud | 74% | 78% | 68% |
| Mem0 | 71% | 77% | 63% |
| Zep Cloud | 61% | 82% | 30% |
| Cognee | 57% | 67% | 43% |
