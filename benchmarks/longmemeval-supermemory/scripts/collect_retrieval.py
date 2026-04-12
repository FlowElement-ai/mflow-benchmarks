#!/usr/bin/env python3
"""
Collect retrieval results from Supermemory for LongMemEval questions.

This script performs search-only queries (no answer generation or judging)
to capture the exact memories retrieved for each question. The output is
merged with existing evaluation results to produce a complete record.

Usage:
    export SUPERMEMORY_API_KEY="sm_..."
    python collect_retrieval.py --max-questions 100 --start-from 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
SUPERMEMORY_BASE_URL = os.environ.get(
    "SUPERMEMORY_BASE_URL", "https://api.supermemory.ai"
)


def _headers():
    return {
        "Authorization": f"Bearer {SUPERMEMORY_API_KEY}",
        "Content-Type": "application/json",
    }


def load_questions(path: Path, max_q: int, start: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    end = min(start + max_q, len(data))
    return data[start:end]


def retrieve_memories(question: str, question_id: str, top_k: int = 10) -> Dict:
    """Query Supermemory search API and return full retrieval details."""
    container_tag = f"lme_{question_id}"
    body = {
        "q": question,
        "containerTag": container_tag,
        "searchMode": "hybrid",
        "limit": top_k,
        "threshold": 0.3,
    }

    start = time.time()
    try:
        resp = requests.post(
            f"{SUPERMEMORY_BASE_URL}/v4/search",
            headers=_headers(),
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.time() - start) * 1000

        results = data.get("results", [])
        memories = []
        for r in results:
            memory = r.get("memory", "")
            if memory:
                memories.append(memory)
            else:
                chunk = r.get("chunk", "")
                if chunk:
                    memories.append(chunk[:500])

        return {
            "retrieved_memories": memories[:top_k],
            "memories_count": len(memories[:top_k]),
            "retrieval_ms": round(elapsed_ms, 2),
            "raw_result_count": len(results),
        }

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {
            "retrieved_memories": [],
            "memories_count": 0,
            "retrieval_ms": round(elapsed_ms, 2),
            "retrieval_error": str(e),
            "raw_result_count": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Collect Supermemory retrieval results")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if not SUPERMEMORY_API_KEY:
        print("Error: set SUPERMEMORY_API_KEY environment variable")
        sys.exit(1)

    data_path = Path(args.data_file) if args.data_file else PROJECT_ROOT / "data" / "longmemeval_oracle.json"
    questions = load_questions(data_path, args.max_questions, args.start_from)
    print(f"Collecting retrieval for {len(questions)} questions (from={args.start_from})")

    results = []
    for idx, q in enumerate(questions):
        qid = q["question_id"]
        q_text = q["question"]
        retrieval = retrieve_memories(q_text, qid, top_k=args.top_k)
        retrieval["question_id"] = qid
        results.append(retrieval)
        print(f"  [{args.start_from + idx + 1}] {qid}: {retrieval['memories_count']} memories ({retrieval['retrieval_ms']:.0f}ms)")
        time.sleep(0.3)

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / "retrieval_collected.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(results), "top_k": args.top_k, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
