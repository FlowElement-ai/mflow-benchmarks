#!/usr/bin/env python3
"""
Zep Cloud LongMemEval Evaluation Script

Evaluates Zep Cloud's knowledge graph memory against the LongMemEval benchmark.

Pipeline:
  1. download  - Download LongMemEval dataset from Google Drive
  2. ingest    - Ingest conversation sessions into Zep Cloud
  3. evaluate  - Run retrieval + LLM answer + LLM judge
  4. baseline  - (Optional) Full-context baseline without Zep

Usage:
  python scripts/zep_evaluate.py download
  python scripts/zep_evaluate.py ingest
  python scripts/zep_evaluate.py evaluate
  python scripts/zep_evaluate.py evaluate --start 0 --num 100
  python scripts/zep_evaluate.py evaluate --detailed-first-n 100
  python scripts/zep_evaluate.py baseline
  python scripts/zep_evaluate.py config
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from time import time

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from zep_cloud import Message
from zep_cloud.client import AsyncZep

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "default_config.json"
)


def load_config(path: str | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(cfg_path):
        print(f"[WARN] Config not found at {cfg_path}, using built-in defaults")
        return {}
    with open(cfg_path) as f:
        return json.load(f)


CFG = load_config()

EDGES_LIMIT: int = CFG.get("edges_limit", 7)
NODES_LIMIT: int = CFG.get("nodes_limit", 3)
EDGES_RERANKER: str = CFG.get("edges_reranker", "cross_encoder")
NODES_RERANKER: str = CFG.get("nodes_reranker", "rrf")
QUERY_MAX_LEN: int = CFG.get("query_max_len", 255)

RESPONSE_MODEL: str = CFG.get("response_model", "gpt-5-mini")
JUDGE_MODEL: str = CFG.get("judge_model", "gpt-4o-mini")

# gpt-5-mini does not support temperature parameter; only default (1) is allowed.
# We track which models need temperature=0 explicitly.
RESPONSE_TEMPERATURE: float | None = CFG.get("response_temperature", None)
JUDGE_TEMPERATURE: float = CFG.get("judge_temperature", 0)

ZEP_BASE_URL: str = CFG.get("zep_base_url", "https://api.getzep.com/api/v2")
USER_PREFIX: str = CFG.get("user_prefix", "lme_oracle_user_")
SESSION_PREFIX: str = CFG.get("session_prefix", "lme_oracle_session_")

DATASET_PATH: str = CFG.get(
    "dataset_path",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "longmemeval_oracle.json"),
)
NUM_QUESTIONS: int = CFG.get("num_questions", 500)
INGEST_CONTENT_MAX_LEN: int = CFG.get("ingest_content_max_len", 4096)
EVAL_BATCH_SIZE: int = CFG.get("eval_batch_size", 2)
ZEP_TIMEOUT: float = CFG.get("zep_timeout_s", 120.0)

RESULTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# ---------------------------------------------------------------------------
# Client Initialization
# ---------------------------------------------------------------------------

load_dotenv(override=True)

zep = AsyncZep(
    api_key=os.getenv("ZEP_API_KEY"),
    base_url=ZEP_BASE_URL,
    timeout=ZEP_TIMEOUT,
)
oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        print("Run: python scripts/zep_evaluate.py download")
        sys.exit(1)
    df = pd.read_json(DATASET_PATH)
    print(f"[INFO] Loaded dataset: {DATASET_PATH} ({len(df)} questions)")
    return df


# ---------------------------------------------------------------------------
# Step 1: Ingest
# ---------------------------------------------------------------------------

async def ingest(df: pd.DataFrame, start_idx: int = 0):
    num = min(NUM_QUESTIONS, len(df))
    print(f"[INGEST] Ingesting {num} users into Zep (starting from idx={start_idx}) ...")

    for idx in range(start_idx, num):
        sessions = df["haystack_sessions"].iloc[idx]
        dates = df["haystack_dates"].iloc[idx]
        user_id = USER_PREFIX + str(idx)
        thread_id = SESSION_PREFIX + str(idx)

        try:
            await zep.user.add(user_id=user_id)
        except Exception:
            pass
        try:
            await zep.thread.create(thread_id=thread_id, user_id=user_id)
        except Exception:
            pass

        msg_count, errors = 0, 0
        for s_idx, session in enumerate(sessions):
            date_obj = datetime.strptime(
                dates[s_idx] + " UTC", "%Y/%m/%d (%a) %H:%M UTC"
            ).replace(tzinfo=timezone.utc)
            for msg in session:
                try:
                    await zep.thread.add_messages(
                        thread_id=thread_id,
                        messages=[
                            Message(
                                role=msg["role"],
                                content=msg["content"][:INGEST_CONTENT_MAX_LEN],
                                created_at=date_obj.isoformat(),
                            )
                        ],
                    )
                    msg_count += 1
                except Exception as e:
                    errors += 1
                    if errors <= 2:
                        print(f"    [WARN] idx={idx} msg error: {e}")

        err_str = f" (errors={errors})" if errors else ""
        print(f"  [{idx + 1}/{num}] user={user_id}  messages={msg_count}{err_str}")

    print("[INGEST] Done. Wait for Zep graph processing before running evaluate.")


# ---------------------------------------------------------------------------
# Step 2: Evaluate
# ---------------------------------------------------------------------------

CONTEXT_TEMPLATE = """FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges.
# If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""


def format_edge_date_range(edge) -> str:
    valid = edge.valid_at if edge.valid_at else "date unknown"
    invalid = edge.invalid_at if edge.invalid_at else "present"
    return f"{valid} - {invalid}"


def compose_search_context(edges, nodes) -> str:
    facts = [f"  - {e.fact} ({format_edge_date_range(e)})" for e in edges]
    entities = [f"  - {n.name}: {n.summary}" for n in nodes]
    return CONTEXT_TEMPLATE.format(
        facts="\n".join(facts), entities="\n".join(entities)
    )


async def lme_response(context: str, question: str) -> str:
    system = (
        "You are a helpful expert assistant answering questions "
        "from lme_experiment users based on the provided context."
    )
    user = (
        "Your task is to briefly answer the question. You are given the following "
        "context from the previous conversation. If you don't know how to answer "
        "the question, abstain from answering.\n"
        f"<CONTEXT>\n{context}\n</CONTEXT>\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\nAnswer:"
    )
    kwargs: dict = dict(
        model=RESPONSE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if RESPONSE_TEMPERATURE is not None:
        kwargs["temperature"] = RESPONSE_TEMPERATURE

    resp = await oai_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


class Grade(BaseModel):
    is_correct: str = Field(description="yes or no")


JUDGE_PROMPTS = {
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. In addition, "
        "do not penalize off-by-one errors for the number of days. If the question asks for the "
        "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still correct."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, the "
        "response should be considered as correct as long as the updated answer is the required answer."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
        "answer no. The model does not need to reflect all the points in the rubric. The response "
        "is correct as long as it recalls and utilizes the user's personal information correctly."
    ),
    "default": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no."
    ),
}


async def lme_grader(question: str, gold: str, response: str, q_type: str) -> bool:
    template_key = q_type if q_type in JUDGE_PROMPTS else "default"
    intro = JUDGE_PROMPTS[template_key]

    if q_type == "single-session-preference":
        body = (
            f"\n\n<QUESTION>\nB: {question}\n</QUESTION>\n"
            f"<RUBRIC>\n{gold}\n</RUBRIC>\n"
            f"<RESPONSE>\nA: {response}\n</RESPONSE>"
        )
    else:
        body = (
            f"\n\n<QUESTION>\nB: {question}\n</QUESTION>\n"
            f"<CORRECT ANSWER>\n{gold}\n</CORRECT ANSWER>\n"
            f"<RESPONSE>\nA: {response}\n</RESPONSE>"
        )

    resp = await oai_client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert grader that determines if answers to questions match a gold standard answer"},
            {"role": "user", "content": intro + body},
        ],
        response_format=Grade,
        temperature=JUDGE_TEMPERATURE,
    )
    result = resp.choices[0].message.parsed
    if result is None:
        return False
    return result.is_correct.strip().lower() == "yes"


async def evaluate_one(
    df: pd.DataFrame, idx: int, detailed: bool = False
) -> dict:
    """Evaluate a single question. Returns a result dict (no shared mutable state)."""
    user_id = USER_PREFIX + str(idx)
    question_id = df["question_id"].iloc[idx]
    question_type = df["question_type"].iloc[idx]
    question = "(date: " + df["question_date"].iloc[idx] + ") " + df["question"].iloc[idx]
    gold_answer = df["answer"].iloc[idx]

    try:
        start = time()
        edges_resp = await zep.graph.search(
            user_id=user_id, reranker=EDGES_RERANKER,
            query=question[:QUERY_MAX_LEN], scope="edges", limit=EDGES_LIMIT,
        )
        nodes_resp = await zep.graph.search(
            user_id=user_id, reranker=NODES_RERANKER,
            query=question[:QUERY_MAX_LEN], scope="nodes", limit=NODES_LIMIT,
        )
        edges = edges_resp.edges or []
        nodes = nodes_resp.nodes or []
        retrieval_dur = time() - start

        context = compose_search_context(edges, nodes)
        hypothesis = await lme_response(context, question)
        total_dur = time() - start

        grade = await lme_grader(question, str(gold_answer), hypothesis, question_type)

        result = {
            "idx": idx,
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "gold_answer": str(gold_answer),
            "hypothesis": hypothesis,
            "grade": grade,
            "retrieval_duration_s": round(retrieval_dur, 3),
            "total_duration_s": round(total_dur, 3),
            "edges_count": len(edges),
            "nodes_count": len(nodes),
            "context_len": len(context.split()),
        }

        if detailed:
            result["edges"] = [
                {"rank": i, "fact": e.fact,
                 "valid_at": str(e.valid_at) if e.valid_at else None,
                 "invalid_at": str(e.invalid_at) if e.invalid_at else None}
                for i, e in enumerate(edges)
            ]
            result["nodes"] = [
                {"rank": i, "name": n.name, "summary": n.summary}
                for i, n in enumerate(nodes)
            ]
            result["context_text"] = context

        return result

    except Exception as e:
        print(f"  [ERROR] idx={idx} question_id={question_id}: {e}")
        return {
            "idx": idx,
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "gold_answer": str(gold_answer),
            "hypothesis": f"ERROR: {e}",
            "grade": False,
            "retrieval_duration_s": 0,
            "total_duration_s": 0,
            "edges_count": 0,
            "nodes_count": 0,
            "context_len": 0,
        }


async def evaluate(
    df: pd.DataFrame,
    start_idx: int = 0,
    num: int | None = None,
    detailed_first_n: int = 0,
):
    """Run evaluation. Results collected from asyncio.gather return values (ordered)."""
    total = min(num or NUM_QUESTIONS, len(df))

    print("=" * 60)
    print("Zep LongMemEval Evaluation")
    print("=" * 60)
    print(f"  Dataset:        {DATASET_PATH}")
    print(f"  Range:          idx {start_idx} - {total - 1} ({total - start_idx} questions)")
    print(f"  Retrieval:      edges={EDGES_LIMIT} ({EDGES_RERANKER}) + nodes={NODES_LIMIT} ({NODES_RERANKER}) = Top-{EDGES_LIMIT + NODES_LIMIT}")
    print(f"  Response model: {RESPONSE_MODEL} (temperature={'default' if RESPONSE_TEMPERATURE is None else RESPONSE_TEMPERATURE})")
    print(f"  Judge model:    {JUDGE_MODEL} (temperature={JUDGE_TEMPERATURE})")
    print(f"  Detailed first: {detailed_first_n}")
    print("=" * 60)

    all_results: list[dict] = []

    idx = start_idx
    while idx < total:
        batch_end = min(idx + EVAL_BATCH_SIZE, total)
        print(f"\n[EVAL] Processing {idx} - {batch_end - 1} ...")

        batch_results = await asyncio.gather(*[
            evaluate_one(df, i, detailed=(i < detailed_first_n))
            for i in range(idx, batch_end)
        ])

        for r in batch_results:
            all_results.append(r)
            mark = "Y" if r["grade"] else "N"
            print(f"  [{r['idx']:3d}] {mark} | {r['question_type']:<28} | {r['hypothesis'][:50]}...")

        idx = batch_end

    correct = sum(1 for r in all_results if r["grade"])
    total_count = len(all_results)
    accuracy = correct / total_count if total_count > 0 else 0

    type_scores: dict[str, list[bool]] = defaultdict(list)
    for r in all_results:
        type_scores[r["question_type"]].append(r["grade"])

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Overall: {correct}/{total_count} = {accuracy:.1%}")
    for qt, grades in sorted(type_scores.items()):
        c = sum(1 for g in grades if g)
        print(f"    {qt:30s}  {c}/{len(grades)} = {c / len(grades):.1%}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "zep_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[SAVED] {results_path}")

    summary = {
        "benchmark": "LongMemEval",
        "dataset_variant": "oracle",
        "memory_system": "Zep Cloud",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "edges_limit": EDGES_LIMIT,
            "nodes_limit": NODES_LIMIT,
            "top_k_total": EDGES_LIMIT + NODES_LIMIT,
            "edges_reranker": EDGES_RERANKER,
            "nodes_reranker": NODES_RERANKER,
            "response_model": RESPONSE_MODEL,
            "response_temperature": RESPONSE_TEMPERATURE,
            "judge_model": JUDGE_MODEL,
            "judge_temperature": JUDGE_TEMPERATURE,
            "ingest_content_max_len": INGEST_CONTENT_MAX_LEN,
        },
        "results": {
            "overall_accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total_count,
            "by_question_type": {
                qt: {
                    "correct": sum(1 for g in grades if g),
                    "total": len(grades),
                    "accuracy": round(sum(1 for g in grades if g) / len(grades), 4),
                }
                for qt, grades in sorted(type_scores.items())
            },
        },
    }
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {summary_path}")


# ---------------------------------------------------------------------------
# Optional: Baseline
# ---------------------------------------------------------------------------

async def baseline(df: pd.DataFrame):
    num = min(NUM_QUESTIONS, len(df))
    print(f"[BASELINE] Running full-context baseline ({num} questions) ...")

    results: list[dict] = []
    for idx in range(num):
        question_id = df["question_id"].iloc[idx]
        question_type = df["question_type"].iloc[idx]
        question = "(date: " + df["question_date"].iloc[idx] + ") " + df["question"].iloc[idx]
        gold_answer = df["answer"].iloc[idx]

        sessions = df["haystack_sessions"].iloc[idx]
        dates = df["haystack_dates"].iloc[idx]
        context = ""
        for s_idx, session in enumerate(sessions):
            date_obj = datetime.strptime(
                dates[s_idx] + " UTC", "%Y/%m/%d (%a) %H:%M UTC"
            ).replace(tzinfo=timezone.utc)
            for msg in session:
                context += f'{msg["role"]} (date: {date_obj}): {msg["content"]}\n'

        start = time()
        hypothesis = await lme_response(context, question)
        dur = time() - start
        grade = await lme_grader(question, str(gold_answer), hypothesis, question_type)

        results.append({
            "idx": idx, "question_id": question_id, "question_type": question_type,
            "hypothesis": hypothesis, "gold_answer": str(gold_answer),
            "duration_s": round(dur, 3), "grade": grade,
        })
        if (idx + 1) % 10 == 0:
            correct = sum(1 for r in results if r["grade"])
            print(f"  [{idx + 1}/{num}] running accuracy: {correct}/{len(results)}")

    correct = sum(1 for r in results if r["grade"])
    print(f"\n[BASELINE] Accuracy: {correct}/{len(results)} = {correct / len(results):.1%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[SAVED] {path}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

async def download_dataset():
    import gdown
    import tarfile

    file_id = "1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80"
    url = f"https://drive.google.com/uc?id={file_id}"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "longmemeval_data.tar.gz")

    if not os.path.exists(tar_path):
        print("[DOWNLOAD] Downloading LongMemEval dataset ...")
        gdown.download(url, tar_path, quiet=False)
    else:
        print(f"[DOWNLOAD] {tar_path} already exists, skipping")

    print("[DOWNLOAD] Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    print("[DOWNLOAD] Done! Available dataset files:")
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".json"):
            size = os.path.getsize(os.path.join(data_dir, f)) / 1024 / 1024
            print(f"  {f}  ({size:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_config():
    print("=" * 60)
    print("Current Configuration")
    print("=" * 60)
    zep_key = os.getenv("ZEP_API_KEY", "NOT SET")
    oai_key = os.getenv("OPENAI_API_KEY", "NOT SET")
    print(f"  ZEP_API_KEY:    {zep_key[:20]}..." if len(zep_key) > 20 else f"  ZEP_API_KEY:    {zep_key}")
    print(f"  OPENAI_API_KEY: {oai_key[:20]}..." if len(oai_key) > 20 else f"  OPENAI_API_KEY: {oai_key}")
    print(f"  ZEP_BASE_URL:   {ZEP_BASE_URL}")
    print()
    print(f"  Retrieval Top-K:      edges={EDGES_LIMIT} + nodes={NODES_LIMIT} = {EDGES_LIMIT + NODES_LIMIT}")
    print(f"  Edges reranker:       {EDGES_RERANKER}")
    print(f"  Nodes reranker:       {NODES_RERANKER}")
    print(f"  Response model:       {RESPONSE_MODEL} (temperature={'default' if RESPONSE_TEMPERATURE is None else RESPONSE_TEMPERATURE})")
    print(f"  Judge model:          {JUDGE_MODEL} (temperature={JUDGE_TEMPERATURE})")
    print()
    print(f"  Dataset:              {DATASET_PATH}")
    print(f"  Num questions:        {NUM_QUESTIONS}")
    print(f"  Batch size:           {EVAL_BATCH_SIZE}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Zep LongMemEval Benchmark")
    parser.add_argument("command", choices=["ingest", "evaluate", "baseline", "download", "config"])
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--num", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--detailed-first-n", type=int, default=0,
                        help="Include structured edge/node details for the first N questions")
    args = parser.parse_args()

    print_config()

    if args.command == "config":
        return

    df = None
    if args.command in ("ingest", "evaluate", "baseline"):
        df = load_dataset()

    if args.command == "ingest":
        asyncio.run(ingest(df, start_idx=args.start))
    elif args.command == "evaluate":
        asyncio.run(evaluate(df, start_idx=args.start, num=args.num,
                             detailed_first_n=args.detailed_first_n))
    elif args.command == "baseline":
        asyncio.run(baseline(df))
    elif args.command == "download":
        asyncio.run(download_dataset())


if __name__ == "__main__":
    main()
