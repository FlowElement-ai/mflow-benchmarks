# Changes from Original Scripts

The scripts in this directory are adapted from `mflow-benchmarks-main/benchmarks/locomo/` for M-flow 0.3.2.

## Modified File: `search_aligned.py`

Three changes were made to adapt to M-flow 0.3.2's updated Python SDK:

### Change 1: `set_database_global_context_variables` → `set_db_context`
- **File**: `search_aligned.py`, line ~92
- **Reason**: M-flow 0.3.2 renamed this function in `m_flow.context_global_variables`
- **Impact**: Functional equivalent, no behavior change

### Change 2: `get_default_user` → `get_seed_user`
- **File**: `search_aligned.py`, line ~95
- **Reason**: M-flow 0.3.2 renamed this function; import path changed to `m_flow.auth.methods.get_seed_user`
- **Impact**: Functional equivalent, no behavior change

### Change 3: `temperature=0.0` removed
- **File**: `search_aligned.py`, line ~278
- **Reason**: `gpt-5-mini` does not support `temperature` parameter (only default value 1 is allowed)
- **Impact**: Answers are non-deterministic; results vary slightly between runs (±1-2%)

## Unmodified Files

- `prompts.py` — identical to original
- `evaluate_aligned.py` — identical to original
- `metrics.py` — identical to original
