# Zep Cloud LongMemEval Benchmark (GPT-5-mini)

Reproducible benchmark evaluating **Zep Cloud's knowledge graph memory** on the [LongMemEval](https://arxiv.org/abs/2407.15460) dataset (`oracle` variant), using **GPT-5-mini** as the response model and **GPT-4o-mini** as the LLM judge.

## Key Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **61.0%** (61/100) |
| temporal-reasoning | 81.7% (49/60) |
| multi-session | 30.0% (12/40) |

## Configuration

| Parameter | Value |
|-----------|-------|
| Memory System | Zep Cloud (Knowledge Graph) |
| Dataset | LongMemEval `oracle` (answer-relevant sessions only) |
| Top-K Retrieval | edges=7 (cross_encoder) + nodes=3 (rrf) = **10** |
| Response Model | **gpt-5-mini** (default temperature) |
| Judge Model | gpt-4o-mini (temperature=0) |
| Content Max Length | 4096 chars |
| Questions Evaluated | 100 (first 100 of oracle) |

> **Note**: `gpt-5-mini` does not support `temperature=0`. The response model uses the default temperature setting. The judge model uses `temperature=0` for deterministic grading.

## Repository Structure

```
zep_longmemeval_benchmark_v2/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── .gitignore
├── configs/
│   └── default_config.json    # All evaluation parameters
├── scripts/
│   ├── zep_evaluate.py        # Main evaluation pipeline
│   └── analyze_results.py     # Results analysis & report generation
├── data/
│   └── longmemeval_oracle.json   # LongMemEval oracle dataset (500 questions)
└── results/
    ├── zep_oracle_100_detailed.json  # Per-question results with retrieval details
    ├── benchmark_summary.json        # Machine-readable summary
    └── BENCHMARK_REPORT.md           # Human-readable detailed report
```

## Quick Start

### Prerequisites

- Python 3.10+
- Zep Cloud API key ([getzep.com](https://www.getzep.com))
- OpenAI API key

### Setup

```bash
# Clone / unzip this directory
cd zep_longmemeval_benchmark_v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your actual keys
```

### Run Evaluation

```bash
# Step 1: Download dataset (if data/ is empty)
python scripts/zep_evaluate.py download

# Step 2: Ingest conversations into Zep Cloud
python scripts/zep_evaluate.py ingest

# Step 3: Wait for Zep graph processing (a few minutes)

# Step 4: Run evaluation (first 100 questions, with detailed output)
python scripts/zep_evaluate.py evaluate --start 0 --num 100 --detailed-first-n 100

# Step 5: Generate report from results
python scripts/analyze_results.py results/zep_eval_results.json
```

### Verify Existing Results

```bash
# Re-generate report from the included results
python scripts/analyze_results.py results/zep_oracle_100_detailed.json
```

### Configuration

All parameters are in `configs/default_config.json`:

```json
{
  "edges_limit": 7,
  "nodes_limit": 3,
  "response_model": "gpt-5-mini",
  "response_temperature": null,
  "judge_model": "gpt-4o-mini",
  "judge_temperature": 0
}
```

## Evaluation Pipeline

1. **Ingest**: Each question's conversation sessions are loaded into Zep Cloud as a user's message thread. Zep automatically builds a temporal knowledge graph (entities, facts, relationships).

2. **Retrieve**: For each question, the script queries Zep's graph search API with the question text, retrieving the top-K most relevant edges (facts) and nodes (entities).

3. **Respond**: The retrieved context (facts with date ranges + entity summaries) is passed to the response model (GPT-5-mini) along with the question to generate an answer.

4. **Judge**: An LLM judge (GPT-4o-mini) compares the model's answer against the gold standard, using question-type-specific rubrics (e.g., temporal-reasoning allows off-by-one errors for day counts).

## Results Format

Each entry in `zep_oracle_100_detailed.json` contains:

```json
{
  "idx": 0,
  "question_id": "...",
  "question_type": "temporal-reasoning",
  "question": "(date: ...) ...",
  "gold_answer": "...",
  "hypothesis": "Model's response",
  "grade": true,
  "retrieval_duration_s": 1.234,
  "total_duration_s": 5.678,
  "edges_count": 7,
  "nodes_count": 3,
  "edges": [
    {"rank": 0, "fact": "...", "valid_at": "...", "invalid_at": "..."}
  ],
  "nodes": [
    {"rank": 0, "name": "...", "summary": "..."}
  ],
  "context_text": "Full context passed to the response model"
}
```

## Methodology Notes

- **Dataset variant**: `oracle` contains only the conversation sessions relevant to each question's answer — no distractor sessions.
- **Content truncation**: Messages longer than 4096 characters are truncated before ingestion (Zep API limit).
- **Temperature**: GPT-5-mini does not support `temperature=0`; it uses the model's default temperature. GPT-4o-mini judge uses `temperature=0` for deterministic grading.
- **Concurrency**: Evaluation uses batch size of 2 with `asyncio.gather` to balance throughput and API rate limits.
- **Result ordering**: Results are collected from `asyncio.gather` return values (which preserve input order), ensuring deterministic question-to-result mapping.

## License

This benchmark script is provided for research and evaluation purposes. The LongMemEval dataset is subject to its original license. Zep Cloud usage is subject to Zep's terms of service.

## Comparisons

LongMemEval Oracle — First 100 Questions | Answer: gpt-5-mini | Judge: gpt-4o-mini | Top-K: 10

| System | Overall | Temporal (60) | Multi-session (40) |
|--------|:-------:|:-------------:|:------------------:|
| **M-flow** | **89%** | **93%** | **82%** |
| Supermemory Cloud | 74% | 78% | 68% |
| Mem0 | 71% | 77% | 63% |
| Zep Cloud | 61% | 82% | 30% |
