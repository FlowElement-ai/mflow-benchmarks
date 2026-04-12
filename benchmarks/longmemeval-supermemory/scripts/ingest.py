#!/usr/bin/env python3
"""
Supermemory LongMemEval Ingestion Script

Ingests LongMemEval Oracle dataset into Supermemory, with each question
isolated via containerTag for independent evaluation.

Alignment with M-flow benchmark:
  1. Same data source: longmemeval_oracle.json
  2. Same text format: [Session Date: xxx]\\n\\nUSER: ...\\n\\nASSISTANT: ...
  3. Per-question isolation: containerTag = "lme_{question_id}"
  4. Per-session documents with indexing confirmation

Features:
  - Exponential backoff retry on 429 rate limits (up to 8 attempts)
  - Document status polling with configurable timeout
  - Incremental ingestion via --start-from
  - Retry failed questions via --retry-ids-file

Usage:
    export SUPERMEMORY_API_KEY="sm_..."
    python ingest.py --max-questions 500 --start-from 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
SUPERMEMORY_BASE_URL = os.environ.get("SUPERMEMORY_BASE_URL", "https://api.supermemory.ai")

POLL_INTERVAL = 5
POLL_TIMEOUT = 300
RETRY_MAX = 8
RETRY_BASE_DELAY = 10


def find_data_file() -> Path:
    """Locate the LongMemEval Oracle dataset file."""
    candidates = [
        PROJECT_ROOT / "data" / "longmemeval_oracle.json",
        SCRIPT_DIR / "data" / "longmemeval_oracle.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Dataset not found. Expected at: {candidates[0]}")


DATA_PATH = find_data_file()


def load_questions() -> List[Dict]:
    """Load all questions from the dataset."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {DATA_PATH.name}")
    return questions


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {SUPERMEMORY_API_KEY}", "Content-Type": "application/json"}


def format_session(session: List[Dict], date_str: str) -> str:
    """Format a session into text, aligned with M-flow's format_session."""
    lines = [f"[Session Date: {date_str}]"]
    for msg in session:
        lines.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n\n".join(lines)


def add_document(content: str, container_tag: str, metadata: Optional[Dict] = None) -> str:
    """Upload a document with exponential backoff retry on 429."""
    body: Dict = {"content": content, "containerTag": container_tag}
    if metadata:
        body["metadata"] = metadata

    for attempt in range(RETRY_MAX):
        resp = requests.post(f"{SUPERMEMORY_BASE_URL}/v3/documents", headers=_headers(), json=body)
        if resp.status_code == 429:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    Rate limited, retrying in {delay}s (attempt {attempt+1}/{RETRY_MAX})...")
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json()["id"]

    raise RuntimeError(f"Max retries ({RETRY_MAX}) exceeded due to rate limiting")


def get_document_status(doc_id: str) -> str:
    """Check processing status of an uploaded document."""
    resp = requests.get(f"{SUPERMEMORY_BASE_URL}/v3/documents/{doc_id}", headers=_headers())
    resp.raise_for_status()
    return resp.json().get("status", "unknown")


def wait_for_documents(doc_ids: List[str], timeout: int = POLL_TIMEOUT) -> Dict[str, str]:
    """Poll until all documents reach terminal status (done/failed/timeout)."""
    pending = set(doc_ids)
    statuses: Dict[str, str] = {}
    start = time.time()

    while pending and (time.time() - start) < timeout:
        for doc_id in list(pending):
            try:
                status = get_document_status(doc_id)
                if status in ("done", "failed"):
                    pending.discard(doc_id)
                    statuses[doc_id] = status
            except Exception as e:
                print(f"    Warning: status check failed for {doc_id}: {e}")

        if pending:
            elapsed = int(time.time() - start)
            print(f"    Indexing: {len(doc_ids) - len(pending)}/{len(doc_ids)} done ({elapsed}s, {len(pending)} pending)")
            time.sleep(POLL_INTERVAL)

    for doc_id in pending:
        statuses[doc_id] = "timeout"

    return statuses


def ingest_question(question: Dict, question_idx: int, total: int) -> Dict:
    """Ingest all sessions for a single question."""
    question_id = question.get("question_id", f"q_{question_idx}")
    container_tag = f"lme_{question_id}"
    sessions = question.get("haystack_sessions", [])
    dates = question.get("haystack_dates", [])

    if len(sessions) != len(dates):
        return {"question_id": question_id, "status": "error", "error": "session/date count mismatch"}

    total_messages = sum(len(s) for s in sessions)
    start_time = time.time()
    print(f"[{question_idx + 1}/{total}] {question_id}: {len(sessions)} sessions, {total_messages} messages")

    try:
        doc_ids = []
        for i, session in enumerate(sessions):
            text = format_session(session, dates[i])
            doc_id = add_document(text, container_tag, metadata={"sessionId": f"{question_id}-session-{i}", "date": dates[i]})
            doc_ids.append(doc_id)

        statuses = wait_for_documents(doc_ids)
        done = sum(1 for s in statuses.values() if s == "done")
        failed = sum(1 for s in statuses.values() if s == "failed")
        timed_out = sum(1 for s in statuses.values() if s == "timeout")
        elapsed = time.time() - start_time

        status = "success" if failed == 0 and timed_out == 0 else "partial"
        symbol = "✓" if status == "success" else "⚠"
        print(f"  {symbol} {elapsed:.1f}s (done={done}, failed={failed}, timeout={timed_out})")

        return {
            "question_id": question_id, "status": status,
            "sessions": len(sessions), "messages": total_messages,
            "doc_ids": doc_ids, "done": done, "failed": failed, "timeout": timed_out,
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Failed: {e}")
        return {"question_id": question_id, "status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Supermemory LongMemEval Ingestion")
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--retry-ids-file", type=str, default="")
    args = parser.parse_args()

    global SUPERMEMORY_API_KEY
    if args.api_key:
        SUPERMEMORY_API_KEY = args.api_key
    if not SUPERMEMORY_API_KEY:
        print("Error: set SUPERMEMORY_API_KEY environment variable or use --api-key")
        sys.exit(1)

    questions = load_questions()

    if args.retry_ids_file:
        with open(args.retry_ids_file) as f:
            retry_ids = set(json.load(f))
        to_process = [q for q in questions if q.get("question_id") in retry_ids]
        suffix = "retry"
    else:
        end = min(args.start_from + args.max_questions, len(questions))
        to_process = questions[args.start_from:end]
        suffix = f"{args.start_from}_{args.start_from + len(to_process)}"

    print(f"\nIngesting {len(to_process)} questions...\n")
    results = []
    t0 = time.time()
    for idx, q in enumerate(to_process):
        results.append(ingest_question(q, idx, len(to_process)))
    total_elapsed = time.time() - t0

    success = sum(1 for r in results if r["status"] == "success")
    partial = sum(1 for r in results if r["status"] == "partial")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone: {success} success, {partial} partial, {errors} errors in {total_elapsed:.0f}s")

    out = PROJECT_ROOT / "results" / f"ingest_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"success": success, "partial": partial, "errors": errors, "elapsed": round(total_elapsed, 1), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
