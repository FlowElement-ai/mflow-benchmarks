#!/usr/bin/env python3
"""
Supermemory LongMemEval Evaluation Script

Evaluates Supermemory's retrieval and answer quality on the LongMemEval
Oracle dataset using:
  - Retrieval: Supermemory v4/search API (hybrid mode, top_k=10)
  - Answer generation: configurable LLM (default: gpt-5-mini)
  - Judge: configurable LLM (default: gpt-4o-mini)
  - Metrics: LLM-Judge accuracy, BLEU-1, token-level F1

Each question is evaluated independently using its own containerTag
for retrieval isolation.

Usage:
    export SUPERMEMORY_API_KEY="sm_..."
    export OPENAI_API_KEY="sk-..."
    python evaluate.py --max-questions 500 --answer-model gpt-5-mini --judge-model gpt-4o-mini
"""

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests
from openai import OpenAI
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
SUPERMEMORY_BASE_URL = os.environ.get("SUPERMEMORY_BASE_URL", "https://api.supermemory.ai")

# ── Prompts ──────────────────────────────────────────────────────────

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


# ── Data Loading ─────────────────────────────────────────────────────

def find_data_file() -> Path:
    candidates = [PROJECT_ROOT / "data" / "longmemeval_oracle.json", SCRIPT_DIR / "data" / "longmemeval_oracle.json"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Dataset not found at {candidates[0]}")


DATA_PATH = find_data_file()


def load_questions(max_q: int, start: int) -> List[Dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    end = min(start + max_q, len(data))
    return data[start:end]


# ── Metrics ──────────────────────────────────────────────────────────

def calculate_bleu1(prediction: str, reference: str) -> float:
    """BLEU-1 score using NLTK with smoothing method 1."""
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
    try:
        return sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def calculate_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    def tokenize(text):
        return set(str(text).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split())
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    common = pred & ref
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(ref)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_llm_judge(question: str, gold: str, generated: str, client: OpenAI, model: str) -> int:
    """Ask LLM to judge if the generated answer is correct."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold_answer=gold, generated_answer=generated)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        return 1 if result.get("label", "").upper() == "CORRECT" else 0
    except Exception as e:
        print(f"  Judge error: {e}")
        return 0


# ── Retrieval ────────────────────────────────────────────────────────

def _sm_headers():
    return {"Authorization": f"Bearer {SUPERMEMORY_API_KEY}", "Content-Type": "application/json"}


def retrieve_memories(question: str, question_id: str, top_k: int = 10) -> str:
    """Search Supermemory for relevant memories within the question's container."""
    body = {
        "q": question,
        "containerTag": f"lme_{question_id}",
        "searchMode": "hybrid",
        "limit": top_k,
        "threshold": 0.3,
    }
    try:
        resp = requests.post(f"{SUPERMEMORY_BASE_URL}/v4/search", headers=_sm_headers(), json=body, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        memories = []
        for r in results:
            mem = r.get("memory", "")
            if mem:
                memories.append(mem)
            elif r.get("chunk"):
                memories.append(r["chunk"][:500])
        return "\n\n".join(memories[:top_k]) if memories else ""
    except Exception as e:
        print(f"  Retrieval error: {e}")
        return ""


# ── Answer Generation ────────────────────────────────────────────────

def generate_answer(memories: str, question: str, client: OpenAI, model: str) -> str:
    """Generate a concise answer based on retrieved memories."""
    if not memories:
        return "No relevant information found"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ANSWER_PROMPT.format(memories=memories, question=question)}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Generation error: {e}")
        return "Error generating answer"


# ── Single Question Evaluation ───────────────────────────────────────

async def evaluate_single(question: Dict, idx: int, total: int, client: OpenAI, answer_model: str, judge_model: str, top_k: int = 10) -> Dict:
    qid = question["question_id"]
    q_text = question["question"]
    gold = str(question["answer"])

    print(f"[{idx + 1}/{total}] {qid}")

    t0 = time.time()
    t_ret = time.time()
    memories = retrieve_memories(q_text, qid, top_k)
    retrieval_ms = (time.time() - t_ret) * 1000

    t_gen = time.time()
    generated = generate_answer(memories, q_text, client, answer_model)
    generation_ms = (time.time() - t_gen) * 1000

    bleu = calculate_bleu1(generated, gold)
    f1 = calculate_f1(generated, gold)
    llm_score = evaluate_llm_judge(q_text, gold, generated, client, judge_model)
    total_ms = (time.time() - t0) * 1000

    print(f"  Answer: {generated[:60]}{'...' if len(generated) > 60 else ''} | BLEU:{bleu:.3f} F1:{f1:.3f} LLM:{llm_score}")

    return {
        "question_id": qid, "question": q_text, "gold_answer": gold,
        "generated_answer": generated, "memories_count": len(memories.split("\n\n")) if memories else 0,
        "bleu_score": round(bleu, 4), "f1_score": round(f1, 4), "llm_score": llm_score,
        "retrieval_ms": round(retrieval_ms, 2), "generation_ms": round(generation_ms, 2), "total_ms": round(total_ms, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Supermemory LongMemEval Evaluation")
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--answer-model", type=str, default="gpt-5-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--api-key", type=str, default="")
    args = parser.parse_args()

    global SUPERMEMORY_API_KEY
    if args.api_key:
        SUPERMEMORY_API_KEY = args.api_key
    if not SUPERMEMORY_API_KEY:
        print("Error: set SUPERMEMORY_API_KEY"); sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Supermemory LongMemEval Evaluation")
    print(f"{'='*60}")
    print(f"Answer model:  {args.answer_model}")
    print(f"Judge model:   {args.judge_model}")
    print(f"Top-K:         {args.top_k}")
    print(f"{'='*60}")

    questions = load_questions(args.max_questions, args.start_from)
    print(f"Evaluating {len(questions)} questions (from={args.start_from})")

    client = OpenAI()
    results = []
    t0 = time.time()

    for idx, q in enumerate(questions):
        result = await evaluate_single(q, args.start_from + idx, args.start_from + len(questions), client, args.answer_model, args.judge_model, args.top_k)
        results.append(result)
        await asyncio.sleep(0.5)

    elapsed = time.time() - t0
    n = len(results) or 1
    summary = {
        "engine": "supermemory", "total_questions": len(results),
        "answer_model": args.answer_model, "judge_model": args.judge_model, "top_k": args.top_k,
        "llm_accuracy": round(sum(r["llm_score"] for r in results) / n, 4),
        "avg_bleu": round(sum(r["bleu_score"] for r in results) / n, 4),
        "avg_f1": round(sum(r["f1_score"] for r in results) / n, 4),
        "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in results) / n, 2),
        "total_time_s": round(elapsed, 2),
    }

    print(f"\n{'='*60}")
    print(f"Results: LLM-Judge={summary['llm_accuracy']:.1%} BLEU={summary['avg_bleu']:.4f} F1={summary['avg_f1']:.4f}")
    print(f"{'='*60}")

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
