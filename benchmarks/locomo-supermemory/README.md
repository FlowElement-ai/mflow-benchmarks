# Supermemory — LoCoMo Benchmark Results

## Results

**Overall: 64.4% (991/1540)** excluding adversarial

Including adversarial (Cat 5): 65.0% (1291/1986)

**MemScore: 65%**

### Per-Category

| Cat | Type | Correct | Total | Score |
|:---:|------------|:-------:|:-----:|:-----:|
| 4 | Single-hop | 587 | 841 | 69.8% |
| 5 | Adversarial | 300 | 446 | 67.3% |
| 1 | Multi-hop | 167 | 282 | 59.2% |
| 2 | Temporal | 188 | 321 | 58.6% |
| 3 | Open-domain | 49 | 96 | 51.0% |

> **Category mapping**: Numbers follow the dataset (`locomo10.json`), NOT the paper's text ordering. See `config/category_mapping.json` (source: snap-research/locomo Issue #27). Note: MemoryBench internally uses paper-order labels (e.g. "world-knowledge" for Cat 4); the table above uses the corrected LoCoMo dataset numbering.

### Retrieval Metrics

| Metric | Value |
|--------|:-----:|
| Hit@K | 84.5% |
| Precision@K | 17.4% |
| Recall@K | 84.5% |
| F1@K | 26.9% |
| MRR | 0.606 |
| NDCG | 0.632 |
| K | 10 |

## System Configuration

| Component | Value |
|-----------|-------|
| System | Supermemory |
| Framework | MemoryBench |
| Run ID | locomo-sm-test-002 |
| Test date | 2026-04-02 to 2026-04-05 |
| Answer LLM | gpt-5-mini |
| Judge LLM | gpt-4o-mini |
| Top-K | 10 |

## File Structure

```
├── README.md
├── METHODOLOGY.md
├── config/
│   ├── system_config.json
│   └── category_mapping.json
├── scripts/                         # MemoryBench framework (TypeScript)
│   ├── src/benchmarks/locomo/       # LOCOMO benchmark definition
│   ├── src/cli/                     # CLI commands
│   ├── package.json
│   └── README.md
├── results/
│   ├── FINAL_SUMMARY.json           # Summary with correct category labels
│   ├── report.json                  # Full MemoryBench report (359MB, includes all searchResults)
│   ├── report_slim.json             # Slim version for browsing (2MB)
│   ├── checkpoint.json              # Run state
│   ├── per_question/                # 1986 individual result files
│   └── test_001/                    # Small-scale test reference (200 questions)
├── run_log/
│   └── supermemory_bench.log
├── data/
│   └── DATA_SOURCE.md
└── known_issues/
    └── NOTES.md
```

## Comparisons

### Aligned (top-k = 10)

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (7e+3n) | 73.4% | gpt-5-mini | gpt-4o-mini | 10 |
| **Supermemory** | **64.4%** | gpt-5-mini | gpt-4o-mini | 10 |

### With vendor-default retrieval budgets

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| Zep Cloud (20e+20n) | 78.4% | gpt-5-mini | gpt-4o-mini | 40 |
| Mem0ᵍ Cloud (published) | 68.5% | — | — | — |
| Mem0 Cloud (published) | 67.1% | — | — | — |
| Mem0 Cloud (tested) | 50.4% | gpt-5-mini | gpt-4o-mini | 30 |
| **Supermemory** | **64.4%** | gpt-5-mini | gpt-4o-mini | 10 |

Mem0 retrieval budget not disclosed in paper. Published scores from [Mem0 paper](https://arxiv.org/abs/2504.19413); answer/judge LLMs not disclosed.

All scores exclude adversarial questions (Cat 5) for comparability.
