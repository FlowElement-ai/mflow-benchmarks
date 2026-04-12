# M-flow LoCoMo Benchmark — Methodology

## 1. Pipeline Overview

```
Step 1: Ingest     Step 2: Search      Step 3: Answer      Step 4: Evaluate
(10h 39m)          (~30min/conv)       (parallel w/ search) (~3min/conv)
                                       
locomo10.json →    For each question:  gpt-5-mini          gpt-4o-mini
M-flow Docker      Retrieve top-10     generates answer     judges CORRECT/WRONG
API ingest         Episode memories    from memories        
                   via EpisodicRetriever
```

## 2. Ingestion

- **Script**: `scripts_original/run_ingest_batched.py`
- **Strategy**: 3-3-4 batching (3 parallel workers per batch)
- **Parameters**: `--no-prune --force`
- **Episode Routing**: disabled (`enable_episode_routing: False` in script payload)
- **Precise Mode**: enabled (`precise_mode: True` in script payload). This is **required** to reproduce the benchmark scores — without it, summarization uses lossy compression and scores drop significantly.
- **Total time**: 10h 39m (272 sessions across 10 conversations)

### Version Requirement

M-flow 0.3.2 (PyPI) has a bug where `config` is undefined in `_task_generate_facets()`, causing all Episode summaries to degrade to raw text truncated at 500 chars. This is fixed in the main branch (commit `3afcb94`). Use the latest main branch to reproduce.

**Verification**: Container logs should show `Generated N sections, M facets` (not `summarize_by_event failed: name 'config' is not defined`).

## 3. Search + Answer

- **Script**: `scripts/search_aligned.py` (adapted for M-flow 0.3.2)
- **Retrieval**: `EpisodicRetriever` with `top_k=10`, `wide_search_top_k=30`
- **Answer LLM**: `gpt-5-mini` (temperature=1, model default — not configurable)
- **Max tokens**: not set (model default)

### Critical: Kuzu File Lock

M-flow uses KuzuDB (embedded graph database) which does NOT support concurrent access. The search script and the Gunicorn API server cannot run simultaneously.

**Solution**: Stop the M-flow API server before running search. Use `docker run` with the committed image (containing ingested data) instead of `docker exec` on the running container.

From `scripts_original/run_benchmark.sh`:
```bash
# Check no uvicorn running (Kuzu lock)
if pgrep -f "uvicorn.*m_flow" > /dev/null 2>&1; then
    echo "ERROR: M-flow API server is running. Stop it first (Kuzu file lock)."
    exit 1
fi
```

**Impact of NOT stopping the server**: ~9% of questions return 0 memories (silent retrieval failure), reducing overall score by ~8 percentage points.

## 4. Evaluation

- **Script**: `scripts/evaluate_aligned.py` (unmodified from original)
- **Judge LLM**: `gpt-4o-mini` (temperature=0, deterministic)
- **Prompt**: Mem0's published `ACCURACY_PROMPT` — generous grading (see `scripts/prompts.py`)
- **Metrics**: LLM-Judge (primary), BLEU-1, F1
- **Category 5** excluded (no gold answers)
- **Run command**: `python evaluate_aligned.py --input-file <search_results> --output-file <eval_output> --model gpt-4o-mini`

## 5. Per-Conversation Execution

Each conversation was run independently to avoid Kuzu lock issues:

```bash
docker run --rm \
  -e MODEL="gpt-5-mini" \
  -e EMBEDDING_MODEL="openai/text-embedding-3-small" \
  -e EMBEDDING_DIMENSIONS=1536 \
  ... (full env vars) ...
  --entrypoint /opt/m_flow/.venv/bin/python \
  m_flow_with_data \
  /opt/m_flow/search_aligned.py \
    --data-path /opt/m_flow/locomo10.json \
    --output-path /opt/m_flow/results/mflow_search_conv{N}.json \
    --top-k 10 \
    --max-conversations {N+1}
```

## 6. Timeout Error Handling

### 6.1 Occurrence

gpt-5-mini occasionally times out on OpenAI API calls (~1-5% of questions per conversation).

### 6.2 Handling in This Test

Timeout errors were retried using a separate script that calls gpt-5-mini with the same memories but a **simplified prompt** (missing 4 guidance rules compared to the original).

### 6.3 Prompt Difference

| Aspect | Original Prompt | Retry Prompt |
|--------|:-:|:-:|
| "do not stop at the first match" | ✓ | ✗ |
| Time reference example `(e.g., "last year"...)` | ✓ | ✗ |
| "combine evidence from multiple memories" | ✓ | ✗ |
| "Look for direct mentions and indirect implications" | ✓ | ✗ |
| Speaker names in header | ✓ | ✗ |

### 6.4 Affected Questions

| Conv | Timeout Retried | Note |
|:----:|:---------------:|------|
| 0 | 0 | Clean run |
| 1 | 4 | Simplified retry prompt used |
| 2 | 4 | Simplified retry prompt used |
| 3 | 0 | Clean run |
| 4 | 0 | Clean run |
| 5 | 0 | Clean run |
| 6 | 0 | Clean run |
| 7 | 0 | Clean run |
| 8 | 0 | Run 5 selected (0 errors) |
| 9 | 11 | Simplified retry prompt used |
| **Total** | **19** | **1.2% of 1540 questions** |

### 6.5 Impact

The 19 retried questions used a weaker prompt, potentially **slightly reducing** their accuracy. The packaged test scripts (`scripts/search_aligned.py`) should be updated to use the full prompt for retries to ensure future runs are not affected.

### 6.6 Recommendation for Reproduction

The scripts in `scripts/` should include automatic retry with the **identical** prompt from `answer_question()`. Future testers should implement retry logic that reuses the exact same prompt template, not a simplified version.

## 7. Known Limitations

1. **gpt-5-mini temperature=1**: Results are non-deterministic. Conv 8 variance test shows ±1.3% (std 0.9%) across 5 runs.
2. **19 timeout retries with simplified prompt**: 1.2% of questions may have slightly weaker answers.
3. **No max_tokens set**: Both answer and judge LLM calls use model defaults.
4. **Category mapping**: The dataset's category numbers differ from the paper's text ordering. Always use the mapping from `config/category_mapping.json`.

## 8. Data Fields

Each question in `results/authoritative/FULL_REPORT_conv{N}.json` contains 14 fields:

| Field | Description |
|-------|-------------|
| `conversation_idx` | Conversation number (0-9) |
| `question` | Question text |
| `category` | Category number (1-5) — see category_mapping.json |
| `gold_answer` | Ground truth answer |
| `evidence` | Evidence reference from dataset |
| `adversarial_answer` | Adversarial answer (if any) |
| `num_memories` | Number of memories retrieved |
| `memories` | Full content of each retrieved memory (text + timestamp + score) |
| `mflow_response` | M-flow's generated answer |
| `llm_judge_score` | Judge verdict: 1=CORRECT, 0=WRONG, null=not evaluated (cat5) |
| `bleu_score` | BLEU-1 score |
| `f1_score` | Token-level F1 score |
