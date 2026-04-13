# MFlow Benchmarks

Comparative benchmark suite evaluating episodic memory systems for AI agents.

## Overview

This repository contains three benchmark datasets and evaluation results:

| Benchmark | Questions | Systems Tested | Key Metric |
|-----------|-----------|----------------|------------|
| **LoCoMo-10** | 1,540 | M-flow vs Mem0 Cloud | LLM-Judge Accuracy |
| **LongMemEval** | 50 | M-flow vs Graphiti | LLM-Judge Accuracy |
| **Evolving Events** | 100 | M-flow vs Cognee vs Graphiti | Human-like Correctness |

## Results Summary

### LoCoMo-10 (Multi-Session Conversational QA)

| System | LLM-Judge | BLEU-1 | F1 |
|--------|-----------|--------|-----|
| M-flow | 81.8% | — | — |
| Mem0 Cloud (published) | 66.9% | — | — |
| Mem0 Cloud (tested) | 50.4% | 0.259 | 0.312 |

### LongMemEval (Long-term Memory QA)

| System | LLM-Judge | F1 | BLEU-1 |
|--------|-----------|-----|--------|
| M-flow | 80% | 0.405 | 0.295 |
| Graphiti | 64% | 0.372 | 0.289 |

### Evolving Events (Multi-hop Reasoning)

| Config | M-flow HL | Cognee HL | Graphiti HL |
|--------|-----------|-----------|-------------|
| k=5, gpt-5-mini | 95.8% | 88.6% | 66.3% |
| k=5, gpt-5.4 | 96.4% | 89.6% | 63.0% |
| k=10, gpt-5-mini | 96.8% | 91.2% | 69.2% |
| k=10, gpt-5.4 | 97.7% | 93.0% | 68.4% |

*HL = Human-like Correctness (DirectLLM judge)*

## Directory Structure

```
mflow-benchmarks/
├── LICENSE
├── README.md
├── CONTRIBUTING.md
└── benchmarks/
    ├── locomo/           # LoCoMo-10 benchmark
    │   ├── README.md
    │   └── run_benchmark.sh
    ├── longmemeval/      # LongMemEval benchmark
    │   ├── README.md
    │   ├── LICENSE
    │   ├── config.yaml
    │   ├── scripts/
    │   └── results/
    └── evolving-events/  # Evolving Events benchmark
        ├── README.md
        └── results/
```

## Quick Start

### LoCoMo-10

```bash
cd benchmarks/locomo
bash run_benchmark.sh
```

### LongMemEval

```bash
cd benchmarks/longmemeval
pip install -r requirements.txt
bash scripts/run_benchmark.sh
```

### Evolving Events

See `benchmarks/evolving-events/README.md` for detailed instructions.

## Evaluation Methodology

- **LLM-Judge**: Binary CORRECT/WRONG evaluation using GPT-4 class models
- **Human-like Correctness**: DirectLLM judge for semantic accuracy
- **DeepEval Correctness**: GEval-based evaluation
- **BLEU-1 / F1**: Token-level precision and recall metrics

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{mflow_benchmarks_2026,
  title = {MFlow Benchmarks},
  author = {MFlow Team},
  year = {2026},
  url = {https://github.com/mflow-ai/mflow-benchmarks}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
