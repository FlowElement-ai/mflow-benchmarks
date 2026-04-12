# Precise Mode

## Problem

M-flow's default summarization pipeline uses a single LLM call (`gpt-5-nano`) to compress conversation text into sectioned summaries. With the default `summarize_content_text.txt` prompt, this produces aggressive compression (4-18% of input), losing critical details:

- Specific dates and times embedded in subordinate clauses
- User-stated numbers (prices, quantities, durations)
- Speaker attribution (who said what)
- Session date metadata

On the LongMemEval benchmark, the default pipeline achieved **~55% LLM-Judge accuracy** on the first 100 questions.

## Solution: Two-Step Pipeline

`precise_mode=True` replaces the single-prompt summarization with:

### Step 1: JSON Section Routing (Lossless)
- LLM outputs JSON `{"sections": [{"title": "...", "sentence_indices": [0,1]}]}`
- Code assembles original text from indices — **zero information loss**
- Session date header injected into each section
- Coverage validation: missing indices appended as "Other" section

### Step 2: Per-Section Concurrent Compression
- Each section compressed independently by LLM (smaller input = better accuracy)
- Sections processed concurrently via `asyncio.gather`
- Compression prompt preserves all dates, numbers, entities, and speaker attribution
- Fallback: if output < 15% of input, original text is kept

### Step 3: Anchor Verification (Code)
- Pre-extracted anchors (dates, numbers, prices, measurements) checked against output
- Missing anchors recovered by inserting the original context at the correct position

## Results

| Metric | Default | Precise Mode | Improvement |
|--------|---------|-------------|-------------|
| LLM-Judge (100q) | ~55% | **89%** | +34 pp |
| BLEU-1 | ~0.21 | **0.30** | +43% |
| F1 | ~0.27 | **0.44** | +63% |

### By Question Type (100 questions)

| Type | Default | Precise |
|------|---------|---------|
| temporal-reasoning | ~50% | **93%** |
| multi-session | ~35% | **82%** |

## Usage

```python
await m_flow.memorize(
    datasets=["my_dataset"],
    content_type=ContentType.DIALOG,
    precise_mode=True,
)
```

Or via HTTP API:
```json
POST /api/v1/memorize
{"datasets": ["my_dataset"], "contentType": "dialog", "preciseMode": true}
```

Or globally via environment variable:
```
MFLOW_PRECISE_MODE=true
```

## Trade-offs

| Aspect | Default | Precise |
|--------|---------|---------|
| LLM calls per event | 1 | 2 + N (N = sections) |
| Compression ratio | 82-96% | 30-50% |
| Storage per question | ~2 KB | ~5-8 KB |
| Ingestion time | ~15 min/q | ~20 min/q |
| Information retention | ~40% of data points | ~100% of data points |
