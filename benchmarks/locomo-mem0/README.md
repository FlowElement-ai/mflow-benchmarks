# Mem0 Cloud — LoCoMo Benchmark Results

## Results

**Overall: 50.4% (776/1540)**

Dataset: LoCoMo-10 (10 conversations, 1540 evaluated questions)

| Metric | Score |
|--------|:-----:|
| LLM-Judge Accuracy | **50.4%** |
| BLEU-1 | 0.259 |
| F1 | 0.312 |

### Per-Category

| Cat | Type | Correct | Total | Score | BLEU-1 | F1 |
|:---:|------------|:-------:|:-----:|:-----:|:------:|:----:|
| 4 | Single-hop | 525 | 841 | 62.4% | 0.311 | 0.379 |
| 1 | Multi-hop | 167 | 282 | 59.2% | 0.241 | 0.339 |
| 3 | Open-domain | 43 | 96 | 44.8% | 0.167 | 0.229 |
| 2 | Temporal | 41 | 321 | 12.8% | 0.164 | 0.135 |

> **Category mapping**: Numbers follow the dataset (`locomo10.json`), NOT the paper's text ordering. See `config/category_mapping.json` for details and source (snap-research/locomo Issue #27).

> **Temporal (Cat 2) scored 12.8%** — far below Mem0's published 55.5%. Mem0's memory system frequently replaces original timestamps with the current date (e.g., "7 May 2023" → "April 2026"), leading to systematically incorrect time-related answers. See [mem0ai/mem0#3944](https://github.com/mem0ai/mem0/issues/3944).

### Per-Conversation

| Conv | Speakers | Questions | Score | BLEU-1 | F1 |
|:----:|----------|:---------:|:-----:|:------:|:----:|
| 0 | Caroline ↔ Melanie | 152 | 55.3% | 0.244 | 0.297 |
| 1 | Jon ↔ Gina | 81 | 40.7% | 0.240 | 0.256 |
| 2 | John ↔ Maria | 152 | 53.3% | 0.280 | 0.350 |
| 3 | Joanna ↔ Nate | 199 | 44.7% | 0.251 | 0.285 |
| 4 | Tim ↔ John | 178 | 53.9% | 0.263 | 0.337 |
| 5 | Audrey ↔ Andrew | 123 | 56.9% | 0.312 | 0.376 |
| 6 | James ↔ John | 150 | 46.7% | 0.274 | 0.317 |
| 7 | Deborah ↔ Jolene | 191 | 50.3% | 0.221 | 0.288 |
| 8 | Evan ↔ Sam | 156 | 47.4% | 0.251 | 0.307 |
| 9 | Calvin ↔ Dave | 158 | 52.5% | 0.263 | 0.302 |

## System Configuration

| Component | Value |
|-----------|-------|
| System | Mem0 Cloud Platform API (mem0ai v1.0.10) |
| API endpoint | `https://api.mem0.ai/v2` |
| LLM (answer) | gpt-5-mini |
| LLM (judge) | gpt-4o-mini (temperature=0) |
| Embedding | text-embedding-3-small (Mem0 platform default) |
| Retrieval top-k | 30 |
| Search strategy | Dual search (both speakers separately) |
| Ingestion | Per-session `client.add()` with speaker-specific `user_id` |
| Test date | 2026-04-12 to 2026-04-13 |

Full configuration: `config/system_config.json`

## Evaluation Methodology

- **Judge**: LLM-as-Judge using Mem0's published ACCURACY_PROMPT (generous grading)
- **Metrics**: LLM-Judge (primary), BLEU-1, F1
- **Category 5** (Adversarial): Excluded per standard methodology (446 questions, no gold answers)
- **Prompts**: Identical to Mem0's official `ANSWER_PROMPT` and `ACCURACY_PROMPT`

## Reproduction

### Prerequisites

- Python 3.x
- `pip install mem0ai openai python-dotenv jinja2 tqdm nltk`
- Mem0 Platform API key (from [app.mem0.ai](https://app.mem0.ai))
- OpenAI API key
- `locomo10.json` dataset (see [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo))

### Steps

```bash
# 1. Configure environment
cp scripts/.env.example scripts/.env
# Edit .env with your API keys

# 2. Place dataset
mkdir -p dataset/
# Download locomo10.json from LoCoMo repo and place in dataset/

# 3. Ingest data (all 10 conversations)
cd scripts/
python run_conv0_add.py                    # Conv 0
python run_all_add.py                      # Conv 1-9

# 4. Evaluate (per conversation)
python run_conv_eval.py --conv 0 --top-k 30
python run_conv_eval.py --conv 1 --top-k 30
# ... repeat for conv 2-9
```

## File Structure

```
locomo-mem0/
├── README.md                        # This file
├── LICENSE
├── config/
│   ├── system_config.json           # Full test configuration
│   └── category_mapping.json        # Correct category labels (with source)
├── scripts/
│   ├── run_conv0_add.py             # Conv 0 ingestion
│   ├── run_all_add.py               # Conv 1-9 ingestion
│   ├── run_conv_eval.py             # Full eval pipeline (per conversation)
│   ├── run_conv0_eval.py            # Conv 0 eval (earlier version)
│   └── .env.example                 # Environment variable template
└── results/
    ├── FINAL_SUMMARY.json           # Aggregated metrics with corrected labels
    ├── all_results.json             # Merged per-question results (1540 questions)
    └── per_conversation/            # Raw results per conversation
        ├── conv0_eval.json
        └── conv{1-9}_full_eval.json
```

## Comparisons

### Aligned (top-k = 10)

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (7e+3n) | 73.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

Mem0 Cloud was tested with top-k=30 (vendor default); no top-k=10 aligned run available.

### With vendor-default retrieval budgets

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (20e+20n) | 78.4% | gpt-5-mini | gpt-4o-mini | 40 |
| Mem0ᵍ Cloud (published) | 68.5% | — | — | — |
| Mem0 Cloud (published) | 67.1% | — | — | — |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |
| **Mem0 Cloud (tested)** | **50.4%** | gpt-5-mini | gpt-4o-mini | 30 |

Mem0 published scores from [Mem0 paper](https://arxiv.org/abs/2504.19413); answer/judge LLMs not disclosed.
