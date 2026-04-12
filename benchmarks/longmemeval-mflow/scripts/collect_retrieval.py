#!/usr/bin/env python3
"""
Collect retrieval results from M-flow for LongMemEval questions.

Runs inside the Docker container to query the local EpisodicRetriever
and capture the exact memories text for each question.

Usage (inside container):
    python3 mflow_collect_retrieval.py --max-questions 100 --start-from 0
"""

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path

MFLOW_ROOT = os.environ.get("MFLOW_ROOT", "/opt/m_flow")
sys.path.insert(0, MFLOW_ROOT)
os.environ.setdefault("VECTOR_DB_URL", "")

from dotenv import load_dotenv
load_dotenv(os.path.join(MFLOW_ROOT, ".env"))

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def find_data_file():
    for p in [PROJECT_ROOT / "data" / "longmemeval_oracle.json", SCRIPT_DIR / "data" / "longmemeval_oracle.json"]:
        if p.exists():
            return p
    raise FileNotFoundError("longmemeval_oracle.json not found")


async def retrieve_memories(question, question_id, top_k=10):
    """Retrieve memories from M-flow's local EpisodicRetriever."""
    from m_flow.retrieval.episodic_retriever import EpisodicRetriever
    from m_flow.retrieval.episodic import EpisodicConfig

    try:
        from m_flow.context_global_variables import backend_access_control_enabled, set_db_context
        from m_flow.data.methods import get_datasets_by_name
        from m_flow.auth.methods.get_seed_user import get_seed_user
    except ImportError:
        from m_flow.context_global_variables import backend_access_control_enabled, set_database_global_context_variables as set_db_context
        from m_flow.data.methods import get_datasets_by_name
        from m_flow.auth.methods import get_default_user as get_seed_user

    dataset_name = f"lme_{question_id}"

    try:
        if backend_access_control_enabled():
            user = await get_seed_user()
            datasets = await get_datasets_by_name(dataset_name, user.id)
            if datasets:
                await set_db_context(datasets[0].id, datasets[0].owner_id)
            else:
                return "", 0, 0.0

        config = EpisodicConfig(top_k=top_k, wide_search_top_k=top_k * 3, display_mode="summary")
        retriever = EpisodicRetriever(config=config)

        t0 = time.time()
        edges = await retriever.get_triplets(question)
        retrieval_ms = (time.time() - t0) * 1000

        if not edges:
            return "", 0, retrieval_ms

        memories = []
        seen = set()
        for edge in edges:
            for node in (getattr(edge, "node1", None), getattr(edge, "node2", None)):
                if node is None:
                    continue
                attrs = getattr(node, "attributes", {}) or {}
                if attrs.get("type") == "Episode":
                    nid = str(getattr(node, "id", ""))
                    if nid in seen:
                        continue
                    seen.add(nid)
                    summary = attrs.get("summary", "")
                    if summary:
                        memories.append(summary)
            edge_name = getattr(edge, "name", None)
            if edge_name and edge_name not in memories:
                memories.append(edge_name)

        text = "\n\n".join(memories[:top_k]) if memories else ""
        return text, len(memories[:top_k]), retrieval_ms

    except Exception as e:
        print(f"  Error: {e}")
        return "", 0, 0.0


async def main():
    parser = argparse.ArgumentParser(description="Collect M-flow retrieval results")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    data_path = find_data_file()
    with open(data_path) as f:
        questions = json.load(f)
    end = min(args.start_from + args.max_questions, len(questions))
    to_process = questions[args.start_from:end]
    print(f"Collecting retrieval for {len(to_process)} questions")

    results = []
    for idx, q in enumerate(to_process):
        qid = q["question_id"]
        text, count, ms = await retrieve_memories(q["question"], qid)
        results.append({
            "question_id": qid,
            "retrieved_memories": text,
            "memories_count": count,
            "retrieval_ms": round(ms, 2),
        })
        print(f"  [{args.start_from + idx + 1}] {qid}: {count} memories ({ms:.0f}ms)")

    out_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / "mflow_retrieval_collected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(results), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
