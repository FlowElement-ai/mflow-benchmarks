#!/usr/bin/env python3
"""
Analyze Cognee LongMemEval evaluation results.

Reads the evaluation JSON and prints a formatted summary table,
per-type breakdown, and optionally exports summary files.

Usage:
    python analyze_results.py results/cognee_eval_results_100.json
    python analyze_results.py results/cognee_eval_results_100.json --export
"""

import argparse
import json
import sys
from pathlib import Path


def analyze(results_path: str, export: bool = False):
    path = Path(results_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if "summary" in data and "results" in data:
        summary = data["summary"]
        results = data["results"]
    elif isinstance(data, list):
        results = data
        summary = None
    else:
        print("ERROR: Unrecognized results format.")
        sys.exit(1)

    n = len(results)
    if n == 0:
        print("No results found.")
        return

    correct = sum(r.get("llm_score", 0) for r in results)
    avg_bleu = sum(r.get("bleu_score", 0) for r in results) / n
    avg_f1 = sum(r.get("f1_score", 0) for r in results) / n
    avg_ret = sum(r.get("retrieval_ms", 0) for r in results) / n
    avg_gen = sum(r.get("generation_ms", 0) for r in results) / n
    zero_mem = sum(1 for r in results if r.get("memories_count", 0) == 0)

    print(f"\n{'=' * 60}")
    print(f"  Cognee LongMemEval Results Analysis")
    print(f"{'=' * 60}")
    print(f"  File:              {path.name}")
    print(f"  Total questions:   {n}")
    print(f"  LLM-Judge Acc:     {correct}/{n} ({correct / n * 100:.1f}%)")
    print(f"  BLEU-1:            {avg_bleu:.4f}")
    print(f"  F1:                {avg_f1:.4f}")
    print(f"  Avg retrieval:     {avg_ret:.0f}ms")
    print(f"  Avg generation:    {avg_gen:.0f}ms")
    print(f"  Zero-memory Qs:    {zero_mem}/{n}")

    if summary:
        answer_model = summary.get("answer_model", "N/A")
        judge_model = summary.get("judge_model", "N/A")
        print(f"  Answer model:      {answer_model}")
        print(f"  Judge model:       {judge_model}")

    # Per-type breakdown
    type_stats = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if qt not in type_stats:
            type_stats[qt] = {"correct": 0, "total": 0,
                              "bleu": 0.0, "f1": 0.0}
        type_stats[qt]["total"] += 1
        type_stats[qt]["correct"] += r.get("llm_score", 0)
        type_stats[qt]["bleu"] += r.get("bleu_score", 0)
        type_stats[qt]["f1"] += r.get("f1_score", 0)

    print(f"\n  {'Question Type':<30} {'Correct':>8} {'Total':>6} {'Acc':>8} {'BLEU':>8} {'F1':>8}")
    print(f"  {'-' * 70}")
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        acc = s["correct"] / s["total"] if s["total"] else 0
        bleu = s["bleu"] / s["total"] if s["total"] else 0
        f1 = s["f1"] / s["total"] if s["total"] else 0
        print(f"  {qt:<30} {s['correct']:>8} {s['total']:>6} "
              f"{acc * 100:>7.1f}% {bleu:>8.4f} {f1:>8.4f}")
    print(f"{'=' * 60}\n")

    if export:
        out_dir = path.parent

        eval_summary = {
            "engine": "cognee",
            "total_questions": n,
            "correct": correct,
            "llm_accuracy": round(correct / n, 4),
            "avg_bleu": round(avg_bleu, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_retrieval_ms": round(avg_ret, 2),
            "avg_generation_ms": round(avg_gen, 2),
            "zero_memory_questions": zero_mem,
        }
        if summary:
            eval_summary["answer_model"] = summary.get("answer_model")
            eval_summary["judge_model"] = summary.get("judge_model")

        summary_path = out_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Exported: {summary_path}")

        by_type = {}
        for qt, s in type_stats.items():
            acc = s["correct"] / s["total"] if s["total"] else 0
            by_type[qt] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": round(acc, 4),
                "avg_bleu": round(s["bleu"] / s["total"], 4) if s["total"] else 0,
                "avg_f1": round(s["f1"] / s["total"], 4) if s["total"] else 0,
            }
        type_path = out_dir / "eval_by_type.json"
        with open(type_path, "w") as f:
            json.dump(by_type, f, indent=2)
        print(f"Exported: {type_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Cognee LongMemEval results")
    parser.add_argument("results_file", help="Path to eval results JSON")
    parser.add_argument("--export", action="store_true",
                        help="Export eval_summary.json and eval_by_type.json")
    args = parser.parse_args()
    analyze(args.results_file, args.export)


if __name__ == "__main__":
    main()
