#!/usr/bin/env python3
"""
Export detailed evaluation results with FULL (non-truncated) retrieved memories.

The main eval script (mem0_qa_eval.py) truncates memories_retrieved to 500 chars
for storage efficiency. This script re-retrieves memories from Mem0 for specified
questions and merges them with existing evaluation results, producing a detailed
output file suitable for open-source benchmark publication.

All scores (bleu_score, f1_score, llm_score) are preserved from the original
evaluation — only the memories_retrieved field is replaced with the full text.

Usage:
    python export_detailed_results.py --max-questions 100
    python export_detailed_results.py --max-questions 100 --results-file results/mem0_eval_results_500.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _parse_search_response(results, top_k: int) -> list:
    """
    Parse mem0 search() response, compatible with multiple SDK response formats:
      - dict: {"memories": [...]} or {"results": [...]}
      - list: [...] direct list
    """
    memories_list = []
    if isinstance(results, dict):
        memories_list = results.get("memories", results.get("results", []))
    elif isinstance(results, list):
        memories_list = results

    if not memories_list:
        return []

    parsed = []
    for mem in memories_list[:top_k]:
        if isinstance(mem, dict):
            text = mem.get("memory", "")
            if not text:
                continue
            created_at = mem.get("created_at", "")
            if created_at:
                text = f"[{created_at}] {text}"
            parsed.append(text)
        elif isinstance(mem, str) and mem.strip():
            parsed.append(mem.strip())

    return parsed


def retrieve_full_memories(client, question: str, question_id: str, top_k: int = 10) -> str:
    """Retrieve full (non-truncated) memories from Mem0 for a given question."""
    user_id = f"lme_{question_id}"

    for attempt in range(3):
        try:
            results = client.search(
                question,
                filters={"user_id": user_id},
                top_k=top_k,
            )
            parsed = _parse_search_response(results, top_k)
            return "\n\n".join(parsed) if parsed else ""
        except TypeError:
            try:
                results = client.search(
                    question,
                    filters={"user_id": user_id},
                )
                parsed = _parse_search_response(results, top_k)
                return "\n\n".join(parsed) if parsed else ""
            except Exception as e2:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"  Retrieval error (fallback failed): {e2}")
                return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  Retrieval error (attempt {attempt+1}/3): {e}")
            return ""

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Export detailed benchmark results with full retrieved memories"
    )
    parser.add_argument(
        "--max-questions", type=int, default=100,
        help="Number of questions to export (default: 100)"
    )
    parser.add_argument(
        "--results-file", type=str, default="",
        help="Path to existing eval results JSON (default: auto-detect)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of memories to retrieve (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output file path (default: results/mem0_eval_first100_detailed.json)"
    )
    parser.add_argument(
        "--mem0-api-key", type=str, default="",
        help="Mem0 API Key (or set MEM0_API_KEY env var)"
    )
    parser.add_argument(
        "--api-delay", type=float, default=0.3,
        help="Delay between API calls in seconds (default: 0.3)"
    )
    args = parser.parse_args()

    api_key = args.mem0_api_key or os.environ.get("MEM0_API_KEY", "")
    if not api_key:
        print("ERROR: Mem0 API Key required.")
        print("  --mem0-api-key m0-xxx  OR  export MEM0_API_KEY=m0-xxx")
        sys.exit(1)

    results_path = args.results_file
    if not results_path:
        candidates = [
            PROJECT_ROOT / "results" / "mem0_eval_results_100.json",
            PROJECT_ROOT / "results" / "mem0_eval_results_500.json",
            PROJECT_ROOT / "results" / "mem0_eval_results.json",
        ]
        for c in candidates:
            if c.exists():
                results_path = str(c)
                break
        if not results_path:
            print("ERROR: No results file found. Specify with --results-file.")
            sys.exit(1)

    print(f"Loading results from: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = data.get("results", [])
    n = min(args.max_questions, len(all_results))
    target_results = all_results[:n]

    print(f"Exporting {n} questions with full memory retrieval (top_k={args.top_k})")

    from mem0 import MemoryClient
    client = MemoryClient(api_key=api_key)

    try:
        client.search("connectivity test", filters={"user_id": "__export_test__"})
        print("Mem0 connection: OK")
    except Exception as e:
        print(f"Mem0 connection warning: {e}")

    detailed_results = []
    re_retrieved = 0
    kept_original = 0

    for idx, r in enumerate(target_results):
        qid = r["question_id"]
        q_text = r["question"]

        print(f"[{idx+1}/{n}] {qid}", end="")

        full_memories = retrieve_full_memories(client, q_text, qid, top_k=args.top_k)

        entry = dict(r)
        if full_memories:
            entry["memories_retrieved"] = full_memories
            mem_count = len(full_memories.split("\n\n"))
            entry["memories_count"] = mem_count
            re_retrieved += 1
            print(f"  -> {mem_count} memories, {len(full_memories)} chars")
        else:
            kept_original += 1
            print(f"  -> 0 memories (kept original)")

        detailed_results.append(entry)

        if args.api_delay > 0:
            time.sleep(args.api_delay)

    summary = {
        "engine": "mem0",
        "export_type": "detailed_first_100",
        "total_questions": len(detailed_results),
        "top_k": args.top_k,
        "re_retrieved": re_retrieved,
        "zero_memories": sum(1 for r in detailed_results if r["memories_count"] == 0),
        "avg_bleu": round(sum(r["bleu_score"] for r in detailed_results) / n, 4),
        "avg_f1": round(sum(r["f1_score"] for r in detailed_results) / n, 4),
        "llm_accuracy": round(sum(r["llm_score"] for r in detailed_results) / n, 4),
        "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in detailed_results) / n, 2),
        "avg_generation_ms": round(sum(r["generation_ms"] for r in detailed_results) / n, 2),
    }

    output_path = args.output or str(
        PROJECT_ROOT / "results" / f"mem0_eval_first{n}_detailed.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": detailed_results}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Export complete: {output_path}")
    print(f"{'='*60}")
    print(f"  Questions:    {summary['total_questions']}")
    print(f"  Re-retrieved: {re_retrieved}")
    print(f"  Zero memories:{summary['zero_memories']}")
    print(f"  LLM Accuracy: {summary['llm_accuracy']:.4f}")
    print(f"  Avg BLEU-1:   {summary['avg_bleu']:.4f}")
    print(f"  Avg F1:       {summary['avg_f1']:.4f}")


if __name__ == "__main__":
    main()
