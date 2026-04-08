# Bug Fix: phase0a.py `config` Not Defined

> **Status**: This bug has been fixed in the main branch (commit `3afcb94`, 2026-04-06). If you are using a version after 0.3.2, the patch below is no longer needed. It is preserved here so that testers can reproduce the exact environment used for this benchmark run.

## Bug Description

In M-flow 0.3.2, the function `_task_generate_facets()` in `m_flow/memory/episodic/episode_builder/phase0a.py` references a variable `config` (line 327) that is not passed as a parameter.

```python
# Line 327 — config is NOT in the function's parameter list
_precise = getattr(config, "precise_mode", False)  # NameError!
```

## Impact

- `summarize_by_event()` fails for **every** Episode with `NameError: name 'config' is not defined`
- All Episode summaries fall back to raw conversation text truncated at 500 characters
- Episode routing and LLM-based summarization are completely bypassed
- FacetPoint extraction (separate pipeline step) is NOT affected

## Diagnosis

Container logs show 100% fallback rate:
```
[episodic] summarize_by_event failed: name 'config' is not defined, using fallback
```
Repeated for every session across all 272 sessions (0 successes, 1302 fallbacks).

## Fix

Add `config` parameter to `_task_generate_facets()` and pass `ctx.config` from the call site.

See `phase0a_config_fix.patch` for the exact diff.

## Verification

After fix, logs show successful summarization:
```
[episodic] Generated 1 sections, 1 facets, topic='Jon pursuing opening a dance s...'
```

Episode summaries changed from:
- **Before**: Raw text truncated at exactly 500 chars (all identical length)
- **After**: LLM-generated summaries with `【title】content` format, avg 1205 chars (168-2782 range)
