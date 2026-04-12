#!/usr/bin/env python3
"""
Cognee LongMemEval Ingestion Script

Ingests LongMemEval Oracle data into a locally-deployed Cognee instance.
Each question gets its own isolated dataset (lme_{question_id}).
Text is pre-chunked at ~2000 characters respecting sentence boundaries,
then uploaded via the Cognee REST API and processed with cognify.

Usage:
    export OPENAI_API_KEY=sk-...
    python cognee_ingest.py --max-questions 100

Requires a running Cognee Docker instance (see docker/ directory).
"""

import argparse
import bisect
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

COGNEE_URL = os.environ.get("COGNEE_LOCAL_URL", "http://localhost:8001")
COGNEE_EMAIL = os.environ.get("COGNEE_EMAIL", "benchmark@gmail.com")
COGNEE_PASSWORD = os.environ.get("COGNEE_PASSWORD", "Benchmark2026!")

PROGRESS_FILE = PROJECT_ROOT / "results" / "cognee_ingest_progress.json"


# ---------------------------------------------------------------------------
# Data loading & formatting
# ---------------------------------------------------------------------------

def find_data_file() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "longmemeval_oracle.json",
        SCRIPT_DIR / "data" / "longmemeval_oracle.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Data file not found. Place longmemeval_oracle.json in:\n  {candidates[0]}"
    )


def load_questions() -> list:
    path = find_data_file()
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Loaded {len(questions)} questions from {path.name}")
    return questions


def _parse_lme_date(date_str: str) -> str:
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return date_str


def format_session_text(session: list, date_str: str, session_idx: int) -> str:
    iso_date = _parse_lme_date(date_str)
    lines = [f"=== Conversation Session {session_idx + 1} "
             f"[Date: {iso_date}] [Original: {date_str}] ===\n"]
    for msg in session:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}\n")
    return "\n".join(lines)


def format_all_sessions(question: dict) -> str:
    sessions = question.get("haystack_sessions", [])
    dates = question.get("haystack_dates", [])
    parts = []
    for idx, session in enumerate(sessions):
        date_str = dates[idx] if idx < len(dates) else "Unknown date"
        parts.append(format_session_text(session, date_str, idx))
    return "\n\n".join(parts)


def chunk_text_by_sentences(text: str, max_chunk_size: int = 2000) -> list:
    """Split text into chunks ≤ max_chunk_size chars, respecting sentence boundaries."""
    if not text or not text.strip():
        return []
    if len(text) <= max_chunk_size:
        return [text]

    split_points = set()
    for m in re.finditer(r"\n\n+", text):
        split_points.add(m.end())
    for m in re.finditer(r"\n", text):
        split_points.add(m.end())
    for m in re.finditer(r"[.!?]\s+", text):
        split_points.add(m.end())
    split_points = sorted(split_points)

    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_chunk_size:
            tail = text[start:]
            if tail.strip():
                chunks.append(tail)
            break
        end = start + max_chunk_size
        idx = bisect.bisect_right(split_points, end) - 1
        best = None
        if idx >= 0 and split_points[idx] > start:
            best = split_points[idx]
        if best is None:
            space_pos = text.rfind(" ", start, end)
            best = (space_pos + 1) if space_pos > start else end
        chunk = text[start:best]
        if chunk.strip():
            chunks.append(chunk)
        start = best

    return chunks if chunks else ([text] if text.strip() else [])


# ---------------------------------------------------------------------------
# Cognee API
# ---------------------------------------------------------------------------

def login(url: str, email: str, password: str) -> dict:
    r = requests.post(
        f"{url}/api/v1/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_progress(prog: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2, ensure_ascii=False)


def ingest_one(url: str, headers: dict, question: dict, chunk_size: int) -> dict:
    qid = question["question_id"]
    ds_name = f"lme_{qid}"

    sessions = question.get("haystack_sessions", [])
    total_msgs = sum(len(s) for s in sessions)
    full_text = format_all_sessions(question)
    chunks = chunk_text_by_sentences(full_text, chunk_size)
    text_kb = len(full_text.encode()) / 1024

    print(f"  {qid}: {len(sessions)} sessions, {total_msgs} msgs, "
          f"{text_kb:.1f}KB -> {len(chunks)} chunks")

    t0 = time.time()

    for ci, chunk in enumerate(chunks):
        files = [("data", (f"{qid}_chunk_{ci:03d}.txt",
                           io.BytesIO(chunk.encode("utf-8")), "text/plain"))]
        r = requests.post(f"{url}/api/v1/add", headers=headers, files=files,
                          data={"datasetName": ds_name}, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(
                f"add chunk {ci} failed: HTTP {r.status_code} - {r.text[:150]}")
        time.sleep(0.3)

    add_time = time.time() - t0

    t1 = time.time()
    r2 = requests.post(
        f"{url}/api/v1/cognify",
        headers={**headers, "Content-Type": "application/json"},
        json={"datasets": [ds_name]},
        timeout=600,
    )
    if r2.status_code != 200:
        raise RuntimeError(
            f"cognify failed: HTTP {r2.status_code} - {r2.text[:150]}")
    cognify_time = time.time() - t1

    return {
        "question_id": qid,
        "dataset_name": ds_name,
        "num_chunks": len(chunks),
        "text_kb": round(text_kb, 2),
        "add_time": round(add_time, 1),
        "cognify_time": round(cognify_time, 1),
        "total_time": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest LongMemEval data into local Cognee")
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between questions (seconds)")
    args = parser.parse_args()

    print("=" * 60)
    print("Cognee LongMemEval Ingestion")
    print("=" * 60)
    print(f"URL:        {COGNEE_URL}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Questions:  {args.max_questions}")

    print("\nLogging in...")
    headers = login(COGNEE_URL, COGNEE_EMAIL, COGNEE_PASSWORD)
    print("  OK")

    prog = load_progress()
    completed = prog["completed"]
    print(f"Existing progress: {len(completed)} completed")

    questions = load_questions()[:args.max_questions]
    remaining = [q for q in questions if q["question_id"] not in completed]
    print(f"Total: {len(questions)}, remaining: {len(remaining)}")

    if not remaining:
        print("\nAll questions already ingested!")
        return

    print()
    start_time = time.time()

    for i, q in enumerate(remaining):
        qid = q["question_id"]
        idx = len(completed) + 1
        print(f"[{idx}/{args.max_questions}]", end=" ", flush=True)

        try:
            result = ingest_one(COGNEE_URL, headers, q, args.chunk_size)
            completed[qid] = result["dataset_name"]
            save_progress({"completed": completed,
                           "failed": prog.get("failed", {})})
            print(f"    OK (add:{result['add_time']}s "
                  f"cognify:{result['cognify_time']}s "
                  f"total:{result['total_time']}s)")
        except Exception as e:
            err = str(e)[:150]
            prog.setdefault("failed", {})[qid] = err
            save_progress(prog)
            print(f"    FAIL: {err}")

        if (i + 1) % 50 == 0:
            try:
                headers = login(COGNEE_URL, COGNEE_EMAIL, COGNEE_PASSWORD)
                print("  [token refreshed]")
            except Exception:
                pass

        if i < len(remaining) - 1:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    done = len(completed)
    failed = len(prog.get("failed", {}))

    print(f"\n{'=' * 60}")
    print(f"Ingestion complete: {done} OK, {failed} failed, "
          f"{elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"{'=' * 60}")

    final_file = PROJECT_ROOT / "results" / "cognee_ingest_final.json"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_file, "w") as f:
        json.dump({
            "total_questions": args.max_questions,
            "completed": done,
            "completed_map": completed,
            "failed": prog.get("failed", {}),
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"Results: {final_file}")


if __name__ == "__main__":
    main()
