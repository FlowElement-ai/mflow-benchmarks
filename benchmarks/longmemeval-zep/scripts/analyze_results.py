#!/usr/bin/env python3
"""
Analyze Zep LongMemEval benchmark results.

Reads the detailed results JSON and produces:
  1. JSON summary  -> results/benchmark_summary.json
  2. Markdown report -> results/BENCHMARK_REPORT.md
  3. Console output

Usage:
  python scripts/analyze_results.py results/zep_oracle_100_detailed.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "default_config.json"
)


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def analyze(results: list[dict], config: dict) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.get("grade"))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    type_stats = {}
    for qt, items in sorted(by_type.items()):
        c = sum(1 for r in items if r.get("grade"))
        type_stats[qt] = {
            "correct": c,
            "total": len(items),
            "accuracy": round(c / len(items), 4) if items else 0,
        }

    avg_retrieval_time = (
        sum(r.get("retrieval_duration_s", 0) for r in results) / total if total else 0
    )
    avg_total_time = (
        sum(r.get("total_duration_s", 0) for r in results) / total if total else 0
    )

    return {
        "benchmark": "LongMemEval",
        "dataset_variant": "oracle",
        "memory_system": "Zep Cloud",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "edges_limit": config.get("edges_limit", 7),
            "nodes_limit": config.get("nodes_limit", 3),
            "top_k_total": config.get("edges_limit", 7) + config.get("nodes_limit", 3),
            "edges_reranker": config.get("edges_reranker", "cross_encoder"),
            "nodes_reranker": config.get("nodes_reranker", "rrf"),
            "response_model": config.get("response_model", "gpt-5-mini"),
            "response_temperature": config.get("response_temperature", None),
            "judge_model": config.get("judge_model", "gpt-4o-mini"),
            "judge_temperature": config.get("judge_temperature", 0),
            "ingest_content_max_len": config.get("ingest_content_max_len", 4096),
        },
        "results": {
            "overall_accuracy": round(correct / total, 4) if total else 0,
            "correct": correct,
            "total": total,
            "by_question_type": type_stats,
        },
        "timing": {
            "avg_retrieval_s": round(avg_retrieval_time, 3),
            "avg_total_s": round(avg_total_time, 3),
        },
    }


def generate_report(summary: dict, results: list[dict]) -> str:
    cfg = summary["config"]
    res = summary["results"]
    timing = summary.get("timing", {})
    lines: list[str] = []

    lines.append("# Zep Cloud LongMemEval Benchmark Report")
    lines.append("")
    lines.append(f"> Generated: {summary['timestamp']}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Memory System | Zep Cloud (Knowledge Graph) |")
    lines.append(f"| Dataset | LongMemEval `oracle` variant |")
    lines.append(f"| Top-K Retrieval | edges={cfg['edges_limit']} + nodes={cfg['nodes_limit']} = {cfg['top_k_total']} |")
    lines.append(f"| Edge Reranker | {cfg['edges_reranker']} |")
    lines.append(f"| Node Reranker | {cfg['nodes_reranker']} |")
    temp_str = "default" if cfg.get("response_temperature") is None else str(cfg["response_temperature"])
    lines.append(f"| Response Model | {cfg['response_model']} (temperature={temp_str}) |")
    lines.append(f"| Judge Model | {cfg['judge_model']} (temperature={cfg['judge_temperature']}) |")
    lines.append(f"| Content Max Length | {cfg['ingest_content_max_len']} chars |")
    lines.append("")

    lines.append("## Overall Results")
    lines.append("")
    lines.append(f"**Accuracy: {res['correct']}/{res['total']} = {res['overall_accuracy']:.1%}**")
    lines.append("")

    if timing:
        lines.append(f"- Avg retrieval time: {timing.get('avg_retrieval_s', 0):.3f}s")
        lines.append(f"- Avg total time per question: {timing.get('avg_total_s', 0):.3f}s")
        lines.append("")

    lines.append("## Results by Question Type")
    lines.append("")
    lines.append("| Question Type | Correct | Total | Accuracy |")
    lines.append("|---------------|---------|-------|----------|")
    for qt, stats in sorted(res["by_question_type"].items()):
        lines.append(
            f"| {qt} | {stats['correct']} | {stats['total']} | {stats['accuracy']:.1%} |"
        )
    lines.append("")

    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| # | ID | Type | Grade | Edges | Nodes | Retrieval(s) |")
    lines.append("|---|-----|------|-------|-------|-------|-------------|")
    for r in results:
        g = "PASS" if r.get("grade") else "FAIL"
        qid = str(r.get("question_id", ""))[:12]
        lines.append(
            f"| {r.get('idx', '')} | {qid} | {r.get('question_type', '')} "
            f"| {g} | {r.get('edges_count', 0)} | {r.get('nodes_count', 0)} "
            f"| {r.get('retrieval_duration_s', 0):.2f} |"
        )
    lines.append("")

    failed = [r for r in results if not r.get("grade")]
    if failed:
        lines.append("## Failed Questions Detail")
        lines.append("")
        for r in failed:
            lines.append(f"### Q{r.get('idx', '?')}: {str(r.get('question_id', ''))[:20]}")
            lines.append(f"- **Type**: {r.get('question_type', '')}")
            lines.append(f"- **Question**: {str(r.get('question', ''))[:200]}")
            lines.append(f"- **Gold Answer**: {str(r.get('gold_answer', ''))[:200]}")
            lines.append(f"- **Model Answer**: {str(r.get('hypothesis', ''))[:200]}")
            if r.get("edges"):
                lines.append(f"- **Retrieved Facts** ({r.get('edges_count', 0)}):")
                for e in r.get("edges", []):
                    lines.append(f"  - [{e.get('rank', '?')}] {str(e.get('fact', ''))[:120]}")
            if r.get("nodes"):
                lines.append(f"- **Retrieved Entities** ({r.get('nodes_count', 0)}):")
                for n in r.get("nodes", []):
                    lines.append(f"  - [{n.get('rank', '?')}] {n.get('name', '')}: {str(n.get('summary', ''))[:100]}")
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `analyze_results.py` at {summary['timestamp']}*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze LongMemEval results")
    parser.add_argument("results_file", help="Path to results JSON")
    parser.add_argument("-o", "--output-dir", default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"[ERROR] File not found: {args.results_file}")
        sys.exit(1)

    with open(args.results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"[INFO] Loaded {len(results)} results from {args.results_file}")

    config = load_config()
    summary = analyze(results, config)

    print(f"\n  Overall: {summary['results']['correct']}/{summary['results']['total']} "
          f"= {summary['results']['overall_accuracy']:.1%}")
    for qt, s in sorted(summary["results"]["by_question_type"].items()):
        print(f"    {qt:30s}  {s['correct']}/{s['total']} = {s['accuracy']:.1%}")

    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {summary_path}")

    report = generate_report(summary, results)
    report_path = os.path.join(args.output_dir, "BENCHMARK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SAVED] {report_path}")


if __name__ == "__main__":
    main()
