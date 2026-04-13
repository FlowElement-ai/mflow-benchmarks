# Cognee Cloud — LoCoMo Benchmark Results

## Results

**Overall: 79.4% (1223/1540)**

Dataset: LoCoMo-10 (10 conversations, 1540 evaluated questions)

| Metric | Score |
|--------|:-----:|
| LLM-Judge Accuracy | **79.4%** |
| BLEU-1 | 0.293 |
| F1 | 0.387 |

### Per-Category

| Cat | Type | Correct | Total | Score | BLEU-1 | F1 |
|:---:|------------|:-------:|:-----:|:-----:|:------:|:----:|
| 4 | Single-hop | 706 | 841 | 84.0% | 0.317 | 0.418 |
| 1 | Multi-hop | 222 | 282 | 78.7% | 0.257 | 0.331 |
| 2 | Temporal | 235 | 321 | 73.2% | 0.312 | 0.415 |
| 3 | Open-domain | 60 | 96 | 62.5% | 0.127 | 0.177 |

> **Category mapping**: Numbers follow the dataset (`locomo10.json`), NOT the paper's text ordering. See `config/category_mapping.json` for details and source (snap-research/locomo Issue #27).

### Per-Conversation

| Conv | Speakers | Questions | Score | BLEU-1 | F1 |
|:----:|----------|:---------:|:-----:|:------:|:----:|
| 0 | Caroline ↔ Melanie | 152 | 82.2% | 0.308 | 0.394 |
| 1 | Jon ↔ Gina | 81 | 82.7% | 0.267 | 0.381 |
| 2 | John ↔ Maria | 152 | 78.9% | 0.322 | 0.418 |
| 3 | Joanna ↔ Nate | 199 | 79.4% | 0.314 | 0.385 |
| 4 | Tim ↔ John | 178 | 73.0% | 0.273 | 0.353 |
| 5 | Audrey ↔ Andrew | 123 | 77.2% | 0.295 | 0.397 |
| 6 | James ↔ John | 150 | 82.0% | 0.312 | 0.416 |
| 7 | Deborah ↔ Jolene | 191 | 82.2% | 0.279 | 0.380 |
| 8 | Evan ↔ Sam | 156 | 79.5% | 0.278 | 0.382 |
| 9 | Calvin ↔ Dave | 158 | 78.5% | 0.275 | 0.368 |

## System Configuration

| Component | Value |
|-----------|-------|
| System | Cognee Cloud API v1 |
| Search type | CHUNKS (pure vector retrieval) |
| LLM (answer) | gpt-5-mini |
| LLM (judge) | gpt-4o-mini (temperature=0) |
| Embedding | Cognee Cloud default |
| Retrieval top-k | 10 |
| Ingestion | per-session upload → cognify |
| Chunking | TextChunker (paragraph-based) |

Full configuration: `config/system_config.json`

## Evaluation Methodology

- **Judge**: LLM-as-Judge using Mem0's published ACCURACY_PROMPT (generous grading)
- **Metrics**: LLM-Judge (primary), BLEU-1, F1
- **Category 5** (Adversarial): Excluded per standard methodology (446 questions, no gold answers)
- **Answer model**: gpt-5-mini generates concise (5-6 word) answers from retrieved context
- **Evaluation aligned** with M-flow and Mem0 methodology for fair comparison

See `METHODOLOGY.md` for full details including ingestion pipeline, search configuration, and known limitations.

## Comparisons

### Aligned (top-k = 10)

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| **Cognee Cloud** | **79.4%** | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (7e+3n) | 73.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

### With vendor-default retrieval budgets

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| **Cognee Cloud** | **79.4%** | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (20e+20n) | 78.4% | gpt-5-mini | gpt-4o-mini | 40 |
| Mem0ᵍ Cloud (published) | 68.5% | — | — | — |
| Mem0 Cloud (published) | 67.1% | — | — | — |
| Mem0 Cloud (tested) | 50.4% | gpt-5-mini | gpt-4o-mini | 30 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

Mem0 retrieval budget not disclosed in paper. Published scores from [Mem0 paper](https://arxiv.org/abs/2504.19413); answer/judge LLMs not disclosed.

## File Structure

```
├── README.md                          # This file
├── METHODOLOGY.md                     # Detailed methodology
├── config/
│   ├── system_config.json             # Full system configuration
│   └── category_mapping.json          # Correct category labels (with source)
├── scripts/                           # Complete benchmark scripts (unmodified)
│   ├── run_ingest.py                  # Ingest LoCoMo data into Cognee
│   ├── search_aligned.py              # Search + answer generation
│   ├── evaluate_aligned.py            # BLEU/F1/LLM-Judge evaluation
│   ├── generate_scores.py             # Score aggregation
│   ├── metrics.py                     # Metric implementations
│   ├── prompts.py                     # All prompt definitions
│   ├── run_benchmark.sh               # One-click reproduction
│   └── requirements.txt               # Python dependencies
├── results/
│   ├── FINAL_SUMMARY.json             # Authoritative summary
│   ├── per_conversation/              # Complete per-conversation data
│   │   └── conv{0-9}/
│   │       ├── search.json            # Retrieved memories + generated answers
│   │       ├── eval.json              # BLEU/F1/LLM-Judge scores per question
│   │       └── full_report.json       # Combined report with summary statistics
│   └── logs/                          # Execution logs
│       ├── conv{0-9}_search.log       # Search phase logs
│       └── conv{0-9}_eval.log         # Evaluation phase logs
└── data/
    └── DATA_SOURCE.md                 # Dataset download instructions
```

### Data Fields

Each question in `per_conversation/conv{N}/search.json` contains:

| Field | Description |
|-------|-------------|
| `question` | Question text |
| `answer` | Ground truth answer |
| `category` | Category number (1-5) — see `config/category_mapping.json` |
| `response` | Cognee's generated answer |
| `memories` | Retrieved conversation chunks (text + timestamp + score) |
| `num_memories` | Number of memories retrieved |
| `search_type` | Search type used (CHUNKS) |

Each question in `per_conversation/conv{N}/eval.json` contains:

| Field | Description |
|-------|-------------|
| `question` | Question text |
| `answer` | Ground truth answer |
| `response` | Cognee's generated answer |
| `category` | Category number |
| `bleu_score` | BLEU-1 score |
| `f1_score` | Token-level F1 score |
| `llm_score` | Judge verdict: 1=CORRECT, 0=WRONG |

## Reproduction

### Prerequisites

- Python 3.10+
- Cognee Cloud API key (`COGNEE_API_KEY`)
- OpenAI API key (`OPENAI_API_KEY`)
- LoCoMo-10 dataset (see `data/DATA_SOURCE.md`)

### Steps

1. Set up a working directory with the scripts:
   ```bash
   mkdir cognee-locomo-bench && cd cognee-locomo-bench
   cp -r <this-package>/scripts/* .
   mkdir -p dataset results
   ```

2. Place `locomo10.json` in `dataset/`

3. Create `.env` with your API keys:
   ```
   COGNEE_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the full benchmark:
   ```bash
   bash run_benchmark.sh
   ```

   Or step by step:
   ```bash
   # 1. Ingest (uploads conversations to Cognee Cloud)
   python run_ingest.py --data dataset/locomo10.json --run-id r4

   # 2. Search + Answer (one conversation at a time recommended)
   for i in $(seq 0 9); do
     python search_aligned.py \
       --data-path dataset/locomo10.json \
       --output-path results/conv${i}_search.json \
       --dataset-prefix locomo_r4 \
       --top-k 10 \
       --max-conversations $((i+1))
   done

   # 3. Evaluate
   for i in $(seq 0 9); do
     python evaluate_aligned.py \
       --input-file results/conv${i}_search.json \
       --output-file results/conv${i}_eval.json
   done

   # 4. Generate scores
   python generate_scores.py --input-file results/cognee_eval.json
   ```

Total time: ~8 hours (ingest ~1h, search ~5h, eval ~2h).

## License

Benchmark scripts and results are provided for research purposes. The LoCoMo-10 dataset is from [snap-research/locomo](https://github.com/snap-research/locomo) and is subject to its own license.
