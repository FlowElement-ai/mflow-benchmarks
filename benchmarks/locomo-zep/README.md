# Zep Cloud (Graphiti) — LoCoMo Benchmark Results

## Results

### Round 1: Standard Configuration (20 edges + 20 nodes = 40 items/query)

**Overall: 78.4% (1208/1540)**

| Cat | Type | Correct | Total | Score |
|:---:|------------|:-------:|:-----:|:-----:|
| 4 | Single-hop | 677 | 841 | 80.5% |
| 1 | Multi-hop | 224 | 282 | 79.4% |
| 3 | Open-domain | 72 | 96 | 75.0% |
| 2 | Temporal | 235 | 321 | 73.2% |

### Round 2: Aligned Configuration (7 edges + 3 nodes = 10 items/query)

**Overall: 73.4% (1131/1540)**

Retrieval budget reduced to 10 items to align with M-flow's `top_k=10` for fair comparison.

| Cat | Type | Correct | Total | Score |
|:---:|------------|:-------:|:-----:|:-----:|
| 4 | Single-hop | 650 | 841 | 77.3% |
| 2 | Temporal | 229 | 321 | 71.3% |
| 1 | Multi-hop | 190 | 282 | 67.4% |
| 3 | Open-domain | 62 | 96 | 64.6% |

> **Category mapping**: Numbers follow the dataset (`locomo10.json`), NOT the paper's text ordering. See `config/category_mapping.json` (source: snap-research/locomo Issue #27).

### Per-Conversation

| User | Speakers | Questions | Round 1 | Round 2 |
|:----:|----------|:---------:|:-------:|:-------:|
| 0 | Caroline ↔ Melanie | 152 | 86.8% | 81.6% |
| 1 | Jon ↔ Gina | 81 | 82.7% | 80.2% |
| 2 | John ↔ Maria | 152 | 78.9% | 77.0% |
| 3 | Joanna ↔ Nate | 199 | 73.9% | 70.4% |
| 4 | Tim ↔ John | 178 | 74.2% | 65.7% |
| 5 | Audrey ↔ Andrew | 123 | 78.9% | 70.7% |
| 6 | James ↔ John | 150 | 79.3% | 73.3% |
| 7 | Deborah ↔ Jolene | 191 | 74.3% | 69.6% |
| 8 | Evan ↔ Sam | 156 | 81.4% | 75.6% |
| 9 | Calvin ↔ Dave | 158 | 79.1% | 75.9% |

### Reference: Zep Official Score

**75.13%** — from `getzep/zep-papers` using `gpt-4o-mini` for both answer and judge, `limit=20` for both edges and nodes.

## System Configuration

| Component | Value |
|-----------|-------|
| System | Zep Cloud (managed SaaS, powered by Graphiti) |
| API | `https://api.getzep.com/api/v2` |
| Test date | 2026-04-02 to 2026-04-03 |
| LLM (answer) | gpt-5-mini (temperature=1, not configurable) |
| LLM (judge) | gpt-4o-mini (temperature=0) |
| Embedding | Zep Cloud internal (not configurable) |
| Edge reranker | cross_encoder |
| Node reranker | RRF (Reciprocal Rank Fusion) |
| SDK | zep-cloud (original: 2.22.1, adapted: 3.18.0) |

Full configuration: `config/system_config.json`

## Known Issues (Preserved from Official Scripts)

1. **`blip_captions` typo**: 1226 image descriptions silently dropped during ingestion. See `known_issues/blip_captions_bug.md`.
2. **`williolw23` typo**: Judge prompt contains garbled text. See `known_issues/eval_prompt_typo.md`.
3. **SDK breaking changes**: zep-cloud 3.x renamed `group_id` → `graph_id`. See `known_issues/SDK_COMPATIBILITY.md`.

These bugs were preserved to maintain consistency with the official Zep 75.13% benchmark conditions.

## Reproduction

### Prerequisites
- Zep Cloud API key
- OpenAI API key
- `locomo10.json` dataset (see `data/DATA_SOURCE.md`)
- `pip install 'zep-cloud==2.22.1' openai pandas python-dotenv pydantic requests`

### Steps
1. Create `.env` with `ZEP_API_KEY` and `OPENAI_API_KEY`
2. Run ingestion: `python zep_locomo_ingestion.py`
3. Wait 10-30 minutes for Zep to build knowledge graph
4. Run search: `python zep_locomo_search.py`
5. Run response generation: `python zep_locomo_responses.py`
6. Run evaluation: `python zep_locomo_eval.py`

See `METHODOLOGY.md` for detailed instructions.

## File Structure

```
├── README.md
├── METHODOLOGY.md
├── config/
│   ├── system_config.json
│   └── category_mapping.json
├── scripts/                     # Adapted for zep-cloud 3.x + gpt-5-mini
├── scripts_original/            # Original from getzep/zep-papers (unmodified)
├── results/
│   ├── FINAL_SUMMARY.json
│   ├── round1_20e20n/           # search + responses + grades + FULL_REPORT
│   ├── round2_7e3n/             # search + responses + grades + FULL_REPORT
│   └── zep_official_reference/  # Official 75.13% data from zep-papers
├── known_issues/
│   ├── blip_captions_bug.md
│   ├── eval_prompt_typo.md
│   └── SDK_COMPATIBILITY.md
└── data/
    └── DATA_SOURCE.md
```

## Comparisons

### Aligned (top-k = 10)

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| **Zep Cloud (7e+3n)** | **73.4%** | gpt-5-mini | gpt-4o-mini | 10 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

### With vendor-default retrieval budgets

| System | Score | Answer LLM | Judge LLM | Top-K |
|--------|:-----:|------------|-----------|:-----:|
| M-flow | 81.8% | gpt-5-mini | gpt-4o-mini | 10 |
| Cognee Cloud | 79.4% | gpt-5-mini | gpt-4o-mini | 10 |
| **Zep Cloud (20e+20n)** | **78.4%** | gpt-5-mini | gpt-4o-mini | 40 |
| Zep Official | 75.1% | gpt-4o-mini | gpt-4o-mini | 40 |
| Mem0ᵍ Cloud (published) | 68.5% | — | — | — |
| Mem0 Cloud (published) | 67.1% | — | — | — |
| Mem0 Cloud (tested) | 50.4% | gpt-5-mini | gpt-4o-mini | 30 |
| Supermemory | 64.4% | gpt-5-mini | gpt-4o-mini | 10 |

Mem0 retrieval budget not disclosed in paper. Published scores from [Mem0 paper](https://arxiv.org/abs/2504.19413); answer/judge LLMs not disclosed.
