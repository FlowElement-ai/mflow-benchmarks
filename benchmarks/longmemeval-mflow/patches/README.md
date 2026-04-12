# Precise Mode Patches

These files implement the `precise_mode` feature for M-flow's summarization pipeline.

## File Map

| Patch File | Target Path (relative to M-flow root) | Type |
|-----------|---------------------------------------|------|
| `NEW__precise_summarize.py` | `m_flow/knowledge/summarization/precise_summarize.py` | New |
| `NEW__precise_compress.txt` | `m_flow/llm/prompts/precise_compress.txt` | New |
| `MOD__pipeline_contexts.py` | `m_flow/memory/episodic/episode_builder/pipeline_contexts.py` | Modified |
| `MOD__write_episodic_memories.py` | `m_flow/memory/episodic/write_episodic_memories.py` | Modified |
| `MOD__phase0a.py` | `m_flow/memory/episodic/episode_builder/phase0a.py` | Modified |
| `MOD__summarize_by_event.py` | `m_flow/knowledge/summarization/summarize_by_event.py` | Modified |
| `MOD__memorize.py` | `m_flow/api/v1/memorize/memorize.py` | Modified |
| `MOD__get_memorize_router.py` | `m_flow/api/v1/memorize/routers/get_memorize_router.py` | Modified |
| `MOD__adapter.py` | `m_flow/adapters/graph/kuzu/adapter.py` | Modified |

## Changes Summary

### `precise_summarize.py` (New)
Two-step summarization: LLM JSON routing splits sentences into topic sections (lossless, code-assembled), then per-section concurrent compression with anchor verification fallback.

### `precise_compress.txt` (New)
Compression prompt optimized for factual density with zero information loss on dates, numbers, and named entities.

### `pipeline_contexts.py`
Added `precise_mode: bool = False` to `EpisodeConfig` dataclass.

### `write_episodic_memories.py`
Added `precise_mode` parameter, passed through to `_build_episode_config`.

### `phase0a.py`
Added `config` parameter to `_task_generate_facets`, routes to precise pipeline when `config.precise_mode=True`. Generates `session_date_header` from `reference_date`.

### `summarize_by_event.py`
Added `precise_mode` and `session_date_header` parameters. When `precise_mode=True`, delegates to `precise_summarize_by_event` with automatic fallback to original pipeline on failure.

### `memorize.py`
Extracts `precise_mode` from kwargs (or `MFLOW_PRECISE_MODE` env var), passes to `Task(write_episodic_memories, precise_mode=...)`.

### `get_memorize_router.py`
Added `precise_mode` field to `MemorizePayloadDTO` for HTTP API support.

### `adapter.py`
Bugfix: Added `"duplicated primary key"` to the KuzuDB batch MERGE fallback condition, preventing crashes when UNWIND contains duplicate node IDs.

## Apply

```bash
bash scripts/apply_patches.sh /path/to/mflow-root
```
