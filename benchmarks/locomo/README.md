# M-flow 0.3.2 — LoCoMo Benchmark Results

## Results

**Overall: 81.8% (1261/1541)**

Dataset: LoCoMo-10 (10 conversations, 1541 evaluated questions)

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
| M-flow version | 0.3.2-dev |
| LLM (ingestion) | gpt-5-nano |
| LLM (answer) | gpt-5-mini (temperature=1, not configurable) |
| LLM (judge) | gpt-4o-mini (temperature=0) |
| Embedding | text-embedding-3-small (1536 dim) |
| Retrieval top-k | 10 |
| Precise mode | enabled |
| Episodic routing | enabled |
| Graph DB | KuzuDB |
| Vector DB | LanceDB |

Full configuration: `config/system_config.json`

## Evaluation Methodology

- **Judge**: LLM-as-Judge using Mem0's published ACCURACY_PROMPT (generous grading)
- **Metrics**: LLM-Judge (primary), BLEU-1, F1
- **Category 5** (Adversarial): Excluded per standard methodology (no gold answers)

See `METHODOLOGY.md` for full details including timeout handling, Kuzu lock issue, and script adaptations.

## Bug Fix Applied

M-flow 0.3.2 has a bug in `phase0a.py` where the `config` variable is not passed to `_task_generate_facets()`, causing all Episode summarization to fall back to raw text truncation. This was fixed before ingestion. See `fixes/` for the patch and details.

## Reproduction

### Prerequisites
- M-flow 0.3.2 with `phase0a_config_fix.patch` applied
- Docker
- OpenAI API key (for gpt-5-mini answer generation and gpt-4o-mini judging)
- `locomo10.json` dataset (see `data/DATA_SOURCE.md`)

### Steps
1. Deploy M-flow Docker with fix applied
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
├── fixes/
│   ├── phase0a_config_fix.patch # Config bug fix
│   └── FIX_NOTES.md             # Fix description
└── data/
    └── DATA_SOURCE.md           # Dataset download instructions
```

## Comparisons

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| **M-flow 0.3.2** | **81.8%** | gpt-5-mini | gpt-4o-mini | 10 |
| M-flow (previous) | 76.5% | gpt-4o | gpt-4o-mini | 10 |
| Zep Cloud (20e+20n) | 78.4% | gpt-5-mini | gpt-4o-mini | 40 |
| Zep Cloud (7e+3n) | 73.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Mem0 (published) | 67.1% | — | — | 30×2 |
| Mem0ᵍ (published) | 68.5% | — | — | 30×2 |

Note: Different systems use different answer LLMs, retrieval strategies, and context sizes. Direct comparison requires careful consideration of these factors.
