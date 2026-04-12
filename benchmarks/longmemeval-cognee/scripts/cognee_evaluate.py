#!/usr/bin/env python3
"""
Cognee LongMemEval Evaluation Script

Evaluates Cognee's performance on the LongMemEval Oracle benchmark.
For each question: retrieve memories -> generate answer -> compute metrics.

Prompts and metrics are aligned with the mflow/mem0 evaluation scripts
so that results are directly comparable.

Models:
  - Answer:  gpt-5-mini  (reasoning model; no temperature, uses max_completion_tokens)
  - Judge:   gpt-4o-mini (temperature=0, JSON response)

Usage:
    export OPENAI_API_KEY=sk-...
    python cognee_evaluate.py --max-questions 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests as http_requests
from openai import OpenAI
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

COGNEE_URL = os.environ.get("COGNEE_LOCAL_URL", "http://localhost:8001")
COGNEE_EMAIL = os.environ.get("COGNEE_EMAIL", "benchmark@gmail.com")
COGNEE_PASSWORD = os.environ.get("COGNEE_PASSWORD", "Benchmark2026!")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Prompts — identical to mflow_qa_eval.py
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are an intelligent memory assistant. Answer the question based on the retrieved memories.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay attention to timestamps to determine temporal relationships
3. If memories contain contradictory information, prioritize the most recent
4. Answer should be concise (less than 6 words)

# Retrieved Memories:
{memories}

# Question: {question}

Answer:"""

JUDGE_PROMPT = """Label the answer as 'CORRECT' or 'WRONG'.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Be generous: if the answer captures the same meaning or time period, mark CORRECT.
Return JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


# ---------------------------------------------------------------------------
# Data loading
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


def load_dataset_map() -> dict:
    """Load question_id -> dataset_name mapping from ingestion results."""
    for name in ["cognee_ingest_final.json", "cognee_ingest_progress.json"]:
        path = PROJECT_ROOT / "results" / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            m = data.get("completed_map") or data.get("completed")
            if m:
                return m
    return {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login(url: str, email: str, password: str) -> dict:
    r = http_requests.post(
        f"{url}/api/v1/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------------------------------------------------------------------------
# Metrics — identical to mflow_qa_eval.py
# ---------------------------------------------------------------------------

def calculate_bleu1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    try:
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
    except (LookupError, TypeError):
        pred_tokens = prediction.lower().split()
        ref_tokens = [reference.lower().split()]
    if not pred_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    try:
        return sentence_bleu(
            ref_tokens, pred_tokens, weights=(1, 0, 0, 0),
            smoothing_function=smooth)
    except (ValueError, ZeroDivisionError):
        return 0.0


def calculate_f1(prediction: str, reference: str) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    def tokenize(text):
        t = str(text).lower()
        for c in ".,!?":
            t = t.replace(c, " ")
        return set(t.split())

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def evaluate_llm_judge(
    question: str, gold: str, generated: str,
    client: OpenAI, model: str,
) -> int:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold_answer=gold,
                generated_answer=generated)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        return 1 if result.get("label", "").upper() == "CORRECT" else 0
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0


# ---------------------------------------------------------------------------
# Cognee retrieval
# ---------------------------------------------------------------------------

def _clean_context(text: str) -> str:
    text = text.replace("__node_content_start__", "")
    text = text.replace("__node_content_end__", "")
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s:
            lines.append(line)
        elif lines and lines[-1].strip():
            lines.append("")
    return "\n".join(lines).strip()


def _parse_response(data, top_k: int) -> str:
    if not data:
        return ""
    if isinstance(data, str):
        return _clean_context(data)
    if isinstance(data, list):
        parts = []
        for item in data[:top_k]:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                sr = item.get("search_result", item)
                if isinstance(sr, str):
                    parts.append(sr.strip())
                elif isinstance(sr, dict):
                    t = (sr.get("content") or sr.get("text")
                         or sr.get("summary") or sr.get("name") or str(sr))
                    parts.append(str(t).strip())
                elif isinstance(sr, list):
                    for sub in sr:
                        if isinstance(sub, str):
                            parts.append(sub.strip())
                        elif isinstance(sub, dict):
                            t = (sub.get("content") or sub.get("text")
                                 or sub.get("summary") or sub.get("name")
                                 or str(sub))
                            parts.append(str(t).strip())
        raw = "\n\n".join(parts) if parts else ""
        return _clean_context(raw) if raw else ""
    return _clean_context(str(data).strip())


def retrieve_memories(
    url: str, headers: dict, question: str,
    dataset_name: str, top_k: int = 10,
) -> str:
    payload = {"query": question, "datasets": [dataset_name]}
    for attempt in range(3):
        try:
            resp = http_requests.post(
                f"{url}/api/v1/search",
                headers={**headers, "Content-Type": "application/json"},
                json=payload, timeout=120)
            resp.raise_for_status()
            return _parse_response(resp.json(), top_k)
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  Retrieval error: {e}")
            return ""
    return ""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(
    memories: str, question: str, client: OpenAI, model: str,
) -> str:
    if not memories:
        return "No relevant information found"
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": ANSWER_PROMPT.format(
                memories=memories, question=question)}],
        }
        if "gpt-5" in model or "o1" in model or "o3" in model:
            kwargs["max_completion_tokens"] = 2048
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 50
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Answer generation error: {e}")
        return "Error generating answer"


# ---------------------------------------------------------------------------
# Single-question evaluation
# ---------------------------------------------------------------------------

def evaluate_single(
    q: Dict, idx: int, total: int, url: str, headers: dict,
    ds_map: dict, openai_client: OpenAI,
    answer_model: str, judge_model: str, top_k: int,
) -> Dict:
    qid = q["question_id"]
    ds_name = ds_map.get(qid, f"lme_{qid}")
    q_text = q["question"]
    gold = str(q["answer"])

    print(f"[{idx + 1}/{total}] {qid} (ds={ds_name})")

    t0 = time.time()

    t_r = time.time()
    memories = retrieve_memories(url, headers, q_text, ds_name, top_k)
    retrieval_ms = (time.time() - t_r) * 1000

    t_g = time.time()
    generated = generate_answer(memories, q_text, openai_client, answer_model)
    generation_ms = (time.time() - t_g) * 1000

    bleu = calculate_bleu1(generated, gold)
    f1 = calculate_f1(generated, gold)
    llm_score = evaluate_llm_judge(
        q_text, gold, generated, openai_client, judge_model)

    total_ms = (time.time() - t0) * 1000
    mem_count = len(memories.split("\n\n")) if memories else 0

    print(f"  Answer: {generated[:80]}{'...' if len(generated) > 80 else ''}")
    print(f"  Gold: {gold} | BLEU:{bleu:.3f} F1:{f1:.3f} "
          f"LLM:{'OK' if llm_score else 'WRONG'} | "
          f"{mem_count}mem {retrieval_ms:.0f}ms")

    return {
        "question_id": qid,
        "question": q_text,
        "question_type": q.get("question_type", ""),
        "gold_answer": gold,
        "generated_answer": generated,
        "memories_retrieved": memories or "",
        "dataset_name": ds_name,
        "memories_count": mem_count,
        "bleu_score": round(bleu, 4),
        "f1_score": round(f1, 4),
        "llm_score": llm_score,
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "total_ms": round(total_ms, 2),
    }


# ---------------------------------------------------------------------------
# Intermediate save
# ---------------------------------------------------------------------------

def save_partial(results: list, output_dir: Path, start_time: float):
    try:
        path = output_dir / "cognee_eval_partial.json"
        n = len(results) or 1
        summary = {
            "engine": "cognee",
            "total_questions": len(results),
            "llm_accuracy": round(
                sum(r["llm_score"] for r in results) / n, 4),
            "status": "partial",
            "elapsed": round(time.time() - start_time, 2),
        }
        with open(path, "w") as f:
            json.dump({"summary": summary, "results": results},
                      f, indent=2, ensure_ascii=False)
        print(f"  [saved: {len(results)} -> {path.name}]")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cognee on LongMemEval Oracle")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--answer-model", type=str, default="gpt-5-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to partial results file for resumption")
    args = parser.parse_args()

    openai_key = OPENAI_API_KEY
    if not openai_key:
        print("ERROR: OPENAI_API_KEY is required.")
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)

    print(f"\n{'=' * 60}")
    print("Cognee LongMemEval Evaluation")
    print(f"{'=' * 60}")
    print(f"Cognee URL:    {COGNEE_URL}")
    print(f"Answer model:  {args.answer_model}")
    print(f"Judge model:   {args.judge_model}")
    print(f"Top-K:         {args.top_k}")
    print(f"Questions:     {args.max_questions}")

    print("\nLogging in to Cognee...")
    headers = login(COGNEE_URL, COGNEE_EMAIL, COGNEE_PASSWORD)
    print("  OK")

    print("Verifying OpenAI...")
    try:
        openai_client.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        print(f"  OpenAI OK ({args.judge_model})")
    except Exception as e:
        print(f"  OpenAI failed: {e}")
        sys.exit(1)

    ds_map = load_dataset_map()
    print(f"Dataset map: {len(ds_map)} entries")

    data_path = find_data_file()
    with open(data_path) as f:
        all_questions = json.load(f)
    questions = all_questions[:args.max_questions]
    print(f"Questions: {len(questions)}")

    results: List[Dict] = []
    evaluated_ids: set = set()
    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        evaluated_ids = {r["question_id"] for r in results}
        print(f"Resumed: {len(results)} previous results")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}\n")
    start_time = time.time()

    for idx, q in enumerate(questions):
        qid = q["question_id"]
        if qid in evaluated_ids:
            continue

        result = evaluate_single(
            q, idx, len(questions), COGNEE_URL, headers, ds_map,
            openai_client, args.answer_model, args.judge_model, args.top_k)
        results.append(result)

        if len(results) % 10 == 0:
            save_partial(results, output_dir, start_time)

        if len(results) % 50 == 0:
            try:
                headers = login(COGNEE_URL, COGNEE_EMAIL, COGNEE_PASSWORD)
                print("  [token refreshed]")
            except Exception:
                pass

        if args.delay > 0:
            time.sleep(args.delay)

    total_time = time.time() - start_time

    n = len(results) or 1
    correct = sum(r["llm_score"] for r in results)
    summary = {
        "engine": "cognee",
        "total_questions": len(results),
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "avg_bleu": round(sum(r["bleu_score"] for r in results) / n, 4),
        "avg_f1": round(sum(r["f1_score"] for r in results) / n, 4),
        "llm_accuracy": round(correct / n, 4),
        "avg_retrieval_ms": round(
            sum(r["retrieval_ms"] for r in results) / n, 2),
        "avg_generation_ms": round(
            sum(r["generation_ms"] for r in results) / n, 2),
        "total_time_seconds": round(total_time, 2),
    }

    type_stats: Dict[str, Dict] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if qt not in type_stats:
            type_stats[qt] = {"correct": 0, "total": 0}
        type_stats[qt]["total"] += 1
        type_stats[qt]["correct"] += r["llm_score"]
    for qt in type_stats:
        t = type_stats[qt]
        t["accuracy"] = round(
            t["correct"] / t["total"], 4) if t["total"] > 0 else 0
    summary["per_type_accuracy"] = type_stats

    print(f"\n{'=' * 60}")
    print("Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"LLM-Judge Accuracy: {summary['llm_accuracy']:.4f} "
          f"({correct}/{len(results)})")
    print(f"BLEU-1: {summary['avg_bleu']:.4f}")
    print(f"F1:     {summary['avg_f1']:.4f}")
    print(f"Avg retrieval: {summary['avg_retrieval_ms']:.0f}ms")
    print(f"Avg generation: {summary['avg_generation_ms']:.0f}ms")
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    if type_stats:
        print(f"\nBy question type:")
        for qt, s in sorted(type_stats.items()):
            print(f"  {qt}: {s['correct']}/{s['total']} "
                  f"({s['accuracy'] * 100:.1f}%)")

    zero_mem = sum(1 for r in results if r["memories_count"] == 0)
    if zero_mem:
        print(f"\n[WARN] {zero_mem}/{len(results)} questions had zero memories")

    out_path = output_dir / "cognee_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
