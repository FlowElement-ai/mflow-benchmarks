# Supermemory LongMemEval Benchmark

Benchmark evaluation of [Supermemory](https://supermemory.ai) on the [LongMemEval](https://arxiv.org/abs/2407.15167) Oracle dataset (500 questions).

## Results Summary

| Metric | Value |
|--------|-------|
| **LLM-Judge Accuracy** | **69.0%** (345/500) |
| BLEU-1 | 0.2733 |
| F1 (token-level) | 0.3631 |

### By Question Type

| Type | Questions | Accuracy |
|------|-----------|----------|
| single-session-user | 70 | **94%** |
| multi-session | 133 | **74%** |
| knowledge-update | 78 | 68% |
| temporal-reasoning | 133 | 62% |
| single-session-preference | 30 | 60% |
| single-session-assistant | 56 | 50% |

## Configuration

| Parameter | Value |
|-----------|-------|
| Engine | Supermemory (cloud API) |
| Dataset | LongMemEval Oracle (500 questions) |
| Answer model | gpt-5-mini |
| Judge model | gpt-4o-mini |
| Retrieval top_k | 10 |
| Retrieval mode | hybrid |
| Data isolation | containerTag = `lme_{question_id}` |
| Ingestion format | Per-session documents |

## Reproduction

### Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Quick Run (full 500 questions)

```bash
source .env
bash scripts/run_benchmark.sh
```

### Step-by-Step

```bash
source .env

# Step 1: Ingest (can be done incrementally)
python scripts/ingest.py --max-questions 500

# Step 2: Evaluate
python scripts/evaluate.py --max-questions 500 \
  --answer-model gpt-5-mini --judge-model gpt-4o-mini
```

### Incremental / Resume

```bash
# Resume ingestion from question 200
python scripts/ingest.py --max-questions 300 --start-from 200

# Resume evaluation from question 100
python scripts/evaluate.py --max-questions 400 --start-from 100
```

## File Structure

```
├── scripts/
│   ├── ingest.py              # Ingestion (429 retry, polling, incremental)
│   ├── evaluate.py            # Evaluation (retrieve → answer → judge)
│   ├── collect_retrieval.py   # Standalone retrieval collection
│   └── run_benchmark.sh       # One-shot runner
├── data/
│   └── longmemeval_oracle.json    # LongMemEval Oracle dataset (500 questions)
├── results/
│   ├── eval_500_summary.json      # Aggregate metrics
│   ├── eval_500_per_question.json # Per-question: answer, judge, scores, latency
│   ├── eval_by_type.json          # Metrics grouped by question type
│   ├── ingest_summary.json        # Ingestion status
│   └── config.json                # Full configuration snapshot
├── requirements.txt
├── .env.example
└── README.md
```

## Per-Question Result Fields

Each entry in `eval_500_per_question.json` contains:

| Field | Description |
|-------|-------------|
| `question_id` | Unique question identifier |
| `question_type` | Category (temporal-reasoning, multi-session, etc.) |
| `question` | Full question text |
| `gold_answer` | Ground truth answer |
| `generated_answer` | Model-generated answer |
| `memories_count` | Number of memories retrieved |
| `bleu_score` | BLEU-1 score |
| `f1_score` | Token-level F1 score |
| `llm_score` | LLM Judge result (1=correct, 0=wrong) |
| `retrieval_ms` | Retrieval latency in milliseconds |
| `generation_ms` | Answer generation latency |
| `total_ms` | End-to-end latency |

## Methodology

### Ingestion
- Each question's haystack sessions are ingested as separate documents
- Documents use `containerTag = "lme_{question_id}"` for isolation
- Text format: `[Session Date: {date}]\n\nUSER: ...\n\nASSISTANT: ...`
- 429 rate limits handled with exponential backoff (up to 8 retries)
- Document status polled until `done` (300s timeout)

### Evaluation
- Retrieval: `POST /v4/search` with `containerTag`, `searchMode: hybrid`, `limit: 10`
- Answer generation: single LLM call with concise prompt (< 6 words target)
- Judging: LLM-as-judge with generous criteria (meaning match, not exact match)
- BLEU-1: NLTK sentence_bleu with smoothing method 1
- F1: token-level precision/recall on lowercased words

### Known Limitations
- `gpt-5-mini` is a reasoning model that does not support `temperature=0`. Results may vary slightly across runs.
- Supermemory's search API results can vary over time due to index updates. The `memories_count` recorded during evaluation is authoritative.
- The Oracle dataset provides only the exact sessions needed to answer each question (no distractors), testing information retention rather than retrieval filtering.

## License

The LongMemEval dataset is subject to its original license. See the [LongMemEval paper](https://arxiv.org/abs/2407.15167) for details.

## Comparisons

LongMemEval Oracle — First 100 Questions | Answer: gpt-5-mini | Judge: gpt-4o-mini | Top-K: 10

| System | Overall | Temporal (60) | Multi-session (40) |
|--------|:-------:|:-------------:|:------------------:|
| **M-flow** | **89%** | **93%** | **82%** |
| Supermemory Cloud | 74% | 78% | 68% |
| Mem0 | 71% | 77% | 63% |
| Zep Cloud | 61% | 82% | 30% |
