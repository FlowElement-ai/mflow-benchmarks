# Mem0 LongMemEval Benchmark

Benchmarking [Mem0](https://mem0.ai/) Platform API on the
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) dataset (ICLR 2025) —
a comprehensive evaluation of long-term interactive memory capabilities.

The evaluation uses **gpt-5-mini** for answer generation and **gpt-4o-mini** as the
LLM judge. All other parameters (prompts, metrics, top-k) are aligned with the
MFlow LongMemEval benchmark to ensure fair cross-engine comparison.

## Results

**First 100 questions — Mem0 Platform API (v1.0.10)**

| Metric | Value |
|--------|-------|
| **LLM-Judge Accuracy** | **71.0% (71 / 100)** |
| BLEU-1 | 0.2405 |
| F1 | 0.3406 |
| Answer Model | gpt-5-mini |
| Judge Model | gpt-4o-mini |

### Accuracy by Question Type

| Question Type | Accuracy |
|---------------|----------|
| temporal-reasoning | 76.7% (46 / 60) |
| multi-session | 62.5% (25 / 40) |

## Project Structure

```
mem0_longmemeval_benchmark/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── config.yaml                 # Evaluation configuration
├── requirements.txt            # Python dependencies (pinned versions)
├── .env.example                # Environment variable template
├── .gitignore
├── data/
│   ├── longmemeval_oracle.json         # LongMemEval dataset (500 questions)
│   └── README.md                       # Dataset source & citation
├── scripts/
│   ├── mem0_ingest.py                  # Ingest conversations into Mem0
│   ├── mem0_qa_eval.py                 # Retrieve + answer + score
│   ├── run_mem0_benchmark.sh           # One-click runner (ingest → wait → eval)
│   ├── analyze_results.py              # Result analysis & cross-engine comparison
│   └── export_detailed_results.py      # Export full retrieval text (no truncation)
└── results/
    ├── mem0_eval_results_100.json          # 100-question evaluation results
    └── mem0_eval_first100_detailed.json    # First 100 questions with full memories
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export MEM0_API_KEY=m0-your-api-key
export OPENAI_API_KEY=sk-your-api-key
```

### 3. Run Benchmark

```bash
# Full pipeline: ingest → wait → evaluate (50 questions)
bash scripts/run_mem0_benchmark.sh --questions 50

# Full 500 questions
bash scripts/run_mem0_benchmark.sh --questions 500 --clean

# Evaluate only (data already ingested)
bash scripts/run_mem0_benchmark.sh --skip-ingest --questions 500
```

### 4. Step-by-Step

```bash
# Step 1: Ingest conversation data into Mem0
python scripts/mem0_ingest.py --max-questions 500 --clean --session-delay 2.0

# Step 2: Wait for Mem0 async processing (≥120 seconds recommended)
sleep 120

# Step 3: Evaluate
python scripts/mem0_qa_eval.py --max-questions 500

# Step 4: Export detailed results for first 100 questions
python scripts/export_detailed_results.py --max-questions 100
```

### 5. Analyze Results

```bash
# Single-engine analysis
python scripts/analyze_results.py results/mem0_eval_results_100.json

# Cross-engine comparison
python scripts/analyze_results.py --compare path/to/mflow_results.json results/mem0_eval_results_100.json
```

## Methodology

### Memory Isolation

Each of the 500 LongMemEval questions is assigned an isolated `user_id`
(`lme_{question_id}`) in Mem0, equivalent to MFlow's `dataset_name`. This
ensures zero cross-contamination between questions — each question's haystack
conversations are ingested and retrieved independently.

### Ingestion Pipeline

For each question, the script iterates over `haystack_sessions` and ingests
them into Mem0 via `client.add()`, passing:
- **messages**: `[{role, content}]` pairs (excluding metadata like `has_answer`)
- **user_id**: `lme_{question_id}`
- **timestamp**: Original session date parsed to Unix timestamp
- **metadata**: Session date and index

Mem0's `add()` is always asynchronous (the deprecated `async_mode=False` has no
effect), so the benchmark waits 120 seconds after ingestion for backend processing.

### Evaluation Pipeline

For each question:
1. **Retrieve**: `client.search(question, filters={"user_id": uid}, top_k=10)`
2. **Generate**: OpenAI `gpt-5-mini` with the ANSWER_PROMPT (see `config.yaml`)
3. **Score**: BLEU-1, F1, and LLM-Judge (`gpt-4o-mini` with JUDGE_PROMPT)

All prompts, models, and metric implementations are identical to the MFlow benchmark.

### Metrics

| Metric | Implementation |
|--------|---------------|
| BLEU-1 | `nltk.sentence_bleu` with `weights=(1,0,0,0)`, `SmoothingFunction.method1` |
| F1 | Token-level F1 (lowercased, punctuation removed) |
| LLM-Judge | `gpt-4o-mini`, `temperature=0.0`, JSON response `{"label": "CORRECT"/"WRONG"}` |

## Result File Schema

Each entry in `results[]` contains:

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | string | Unique question identifier |
| `question` | string | Original question text |
| `question_type` | string | One of 6 categories |
| `gold_answer` | string | Ground truth answer |
| `generated_answer` | string | Model-generated answer |
| `memories_retrieved` | string | Retrieved memory text |
| `memories_count` | int | Number of memories retrieved |
| `bleu_score` | float | BLEU-1 score |
| `f1_score` | float | Token-level F1 score |
| `llm_score` | int | LLM-Judge result (1=correct, 0=wrong) |
| `retrieval_ms` | float | Memory retrieval latency (ms) |
| `generation_ms` | float | Answer generation latency (ms) |
| `total_ms` | float | Total per-question latency (ms) |

> **Note**: In `mem0_eval_results_100.json`, the `memories_retrieved` field is
> truncated to 500 characters for storage efficiency. The file
> `mem0_eval_first100_detailed.json` contains the **full, non-truncated** retrieval
> text for the first 100 questions.

## Known Mem0 Platform Limitations

1. **Async-only ingestion**: `mem0.add()` is always asynchronous. The deprecated
   `async_mode=False` parameter has no effect. A post-ingestion wait is required.

2. **Semantic search misses (~10%)**: For some questions, memories are stored
   successfully but `search()` returns 0 results — the query-to-memory vector
   similarity falls below Mem0's internal threshold.

3. **Memory extraction failures (~4%)**: Mem0's backend extracts no memories from
   certain conversations, particularly `single-session-assistant` type content
   (e.g., shift schedules, musical notation) that contains structured data rather
   than personal facts.

## Reproducibility

To reproduce the benchmark results:
1. Use the exact dependency versions in `requirements.txt`
2. Use Mem0 Platform API v1.0.10 (`mem0ai==1.0.10`)
3. Use `gpt-5-mini` for answer generation and `gpt-4o-mini` for judging
4. Follow the ingestion → wait → evaluation pipeline as described above

Note: Results may vary slightly due to Mem0's non-deterministic memory extraction
and OpenAI model updates.

## Citation

If you use this benchmark, please cite the LongMemEval paper:

```bibtex
@inproceedings{wu2025longmemeval,
  title     = {LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author    = {Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=pZiyCaVuti}
}
```

## License

This benchmark code is released under the [MIT License](LICENSE).


## Comparisons

LongMemEval Oracle — First 100 Questions | Answer: gpt-5-mini | Judge: gpt-4o-mini | Top-K: 10

| System | Overall | Temporal (60) | Multi-session (40) |
|--------|:-------:|:-------------:|:------------------:|
| **M-flow** | **89%** | **93%** | **82%** |
| Supermemory Cloud | 74% | 78% | 68% |
| Mem0 | 71% | 77% | 63% |
| Zep Cloud | 61% | 82% | 30% |
| Cognee | 57% | 67% | 43% |
