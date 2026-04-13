# M-flow 0.3.2 — LoCoMo Benchmark Results

## Results

**Overall: 81.8% (1260/1540)**

Dataset: LoCoMo-10 (10 conversations, 1540 evaluated questions)

### Per-Category

| Cat | Type | Correct | Total | Score |
|:---:|------------|:-------:|:-----:|:-----:|
| 4 | Single-hop | 737 | 841 | 87.6% |
| 2 | Temporal | 255 | 321 | 79.4% |
| 1 | Multi-hop | 212 | 282 | 75.2% |
| 3 | Open-domain | 56 | 96 | 58.3% |

> **Category mapping**: Numbers follow the dataset (`locomo10.json`), NOT the paper's text ordering. See `config/category_mapping.json` for details and source (snap-research/locomo Issue #27).

### Per-Conversation

| Conv | Speakers | Questions | Score |
|:----:|----------|:---------:|:-----:|
| 0 | Caroline ↔ Melanie | 152 | 83.6% |
| 1 | Jon ↔ Gina | 82 | 90.2% |
| 2 | John ↔ Maria | 152 | 86.2% |
| 3 | Joanna ↔ Nate | 199 | 78.9% |
| 4 | Tim ↔ John | 178 | 77.5% |
| 5 | Audrey ↔ Andrew | 123 | 79.7% |
| 6 | James ↔ John | 150 | 88.7% |
| 7 | Deborah ↔ Jolene | 191 | 80.6% |
| 8 | Evan ↔ Sam | 156 | 75.6% |
| 9 | Calvin ↔ Dave | 158 | 82.9% |

Conv 8 was run 5 times to measure variance: mean 75.1%, std 0.9%, range 73.7%-76.3%. See `results/conv8_variance/`.

## System Configuration

| Component | Value |
|-----------|-------|
| M-flow version | [0.3.2+](https://github.com/FlowElement-ai/m_flow) (commit `3afcb94` or later) |
| LLM (ingestion) | gpt-5-nano |
| LLM (answer) | gpt-5-mini (temperature=1, not configurable) |
| LLM (judge) | gpt-4o-mini (temperature=0) |
| Embedding | text-embedding-3-small (1536 dim) |
| Retrieval top-k | 10 |
| Precise mode | enabled |
| Episodic routing | disabled |
| Graph DB | KuzuDB |
| Vector DB | LanceDB |

Full configuration: `config/system_config.json`

## Evaluation Methodology

- **Judge**: LLM-as-Judge using Mem0's published ACCURACY_PROMPT (generous grading)
- **Metrics**: LLM-Judge (primary), BLEU-1, F1
- **Category 5** (Adversarial): Excluded per standard methodology (no gold answers)

See `METHODOLOGY.md` for full details including timeout handling, Kuzu lock issue, and script adaptations.

## Version Note

This benchmark was run on M-flow 0.3.2 with a bug fix for `phase0a.py` (config not passed to `_task_generate_facets`). The fix is included in the main branch since commit `3afcb94`. Clone the latest main branch to reproduce.

## Reproduction

### Prerequisites
- M-flow from GitHub main branch (commit `3afcb94` or later)
- Docker
- OpenAI API key (for gpt-5-mini answer generation and gpt-4o-mini judging)
- `locomo10.json` dataset (see `data/DATA_SOURCE.md`)

### Steps
1. Clone M-flow repo and build Docker image
2. Ingest using `scripts_original/run_ingest_batched.py --no-prune --force`
3. **Stop the M-flow API server** (Kuzu file lock — critical!)
4. Run search: `scripts/search_aligned.py --top-k 10` (one conv at a time via `docker run`)
5. Fix any timeout errors (retry with same prompt)
6. Evaluate: `scripts/evaluate_aligned.py --model gpt-4o-mini`

See `METHODOLOGY.md` for detailed instructions.

## File Structure

```
├── README.md                    # This file
├── METHODOLOGY.md               # Detailed methodology
├── config/
│   ├── system_config.json       # Full system configuration
│   └── category_mapping.json    # Correct category labels (with source)
├── scripts/                     # Adapted scripts for M-flow 0.3.2
│   ├── search_aligned.py        # 3 SDK adaptations (see CHANGES.md)
│   ├── evaluate_aligned.py      # Unmodified
│   ├── metrics.py               # Unmodified
│   ├── prompts.py               # Unmodified
│   └── CHANGES.md               # Adaptation details
├── scripts_original/            # Original benchmark scripts (unmodified)
├── results/
│   ├── FINAL_SUMMARY.json       # Authoritative summary with correct labels
│   ├── authoritative/           # 10 FULL_REPORT files (14 fields each)
│   ├── conv8_variance/          # 5-run variance test data
│   └── raw_data/                # All intermediate files preserved
└── data/
    └── DATA_SOURCE.md           # Dataset download instructions
```

## Comparisons

### Aligned (top-k = 10)

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| **M-flow** | **81.8%** | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (7e+3n) | 73.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

### With vendor-default retrieval budgets

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| **M-flow** | **81.8%** | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (20e+20n) | 78.4% | gpt-5-mini | gpt-4o-mini | 40 |
| Mem0ᵍ Cloud (published) | 68.5% | — | — | — |
| Mem0 Cloud (published) | 67.1% | — | — | — |
| Mem0 Cloud (tested) | 50.4% | gpt-5-mini | gpt-4o-mini | 30 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

Mem0 retrieval budget not disclosed in paper. Published scores from [Mem0 paper](https://arxiv.org/abs/2504.19413); answer/judge LLMs not disclosed.

Note: Different systems use different retrieval strategies, context assembly methods, and ingestion pipelines. Direct comparison requires careful consideration of these factors. Full methodology and reproduction scripts are provided in each system's subdirectory.
