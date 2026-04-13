"""Single conversation full pipeline: search + answer + evaluate.
Records ALL data required by LocOMo benchmark spec:
  - Per-question: question, gold answer, predicted answer, category, evidence,
    adversarial_answer, speaker memories (both), memory counts, search times,
    response time, BLEU-1, F1, LLM-Judge score
  - Per-conversation summary
  - Per-category summary
"""

import argparse
import json
import os
import re
import time

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

load_dotenv()

from mem0 import MemoryClient
from mem0.memory.utils import extract_json

# ---------- Prompts (identical to Mem0 official) ----------

ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

# ---------- Metrics ----------

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass


def calc_bleu1(pred, ref):
    if not pred or not ref:
        return 0.0
    try:
        pred_tok = nltk.word_tokenize(pred.lower())
        ref_tok = [nltk.word_tokenize(ref.lower())]
        if not pred_tok:
            return 0.0
        return sentence_bleu(ref_tok, pred_tok, weights=(1, 0, 0, 0),
                             smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0


def calc_f1(pred, ref):
    def tokenize(t):
        return set(str(t).lower().replace(".", " ").replace(",", " ")
                   .replace("!", " ").replace("?", " ").split())
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = p & r
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def llm_judge(question, gold, generated, client, model="gpt-4o-mini"):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ACCURACY_PROMPT.format(
                question=question, gold_answer=gold, generated_answer=generated)}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        label = json.loads(extract_json(raw)).get("label", "WRONG").upper()
        return 1 if label == "CORRECT" else 0, raw
    except Exception as e:
        return 0, f"[JUDGE_ERROR: {e}]"


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single LocOMo conversation")
    parser.add_argument("--conv", type=int, default=1, help="Conversation index (0-9)")
    parser.add_argument("--top-k", type=int, default=30, help="top_k for memory search")
    args = parser.parse_args()

    conv_idx = args.conv
    top_k = args.top_k

    mem0_client = MemoryClient(
        api_key=os.getenv("MEM0_API_KEY"),
        org_id=os.getenv("MEM0_ORGANIZATION_ID"),
        project_id=os.getenv("MEM0_PROJECT_ID"),
    )
    openai_client = OpenAI()
    answer_model = os.getenv("MODEL", "gpt-5-mini")
    judge_model = "gpt-4o-mini"

    with open("dataset/locomo10.json") as f:
        data = json.load(f)

    item = data[conv_idx]
    conv = item["conversation"]
    qa_list = item["qa"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    speaker_a_id = f"{speaker_a}_{conv_idx}"
    speaker_b_id = f"{speaker_b}_{conv_idx}"

    evaluated = [q for q in qa_list if q.get("category") != 5]

    output_file = f"results/conv{conv_idx}_full_eval.json"

    print(f"Conv {conv_idx}: {speaker_a} <-> {speaker_b}")
    print(f"Total QA: {len(qa_list)}, Evaluated (excl cat5): {len(evaluated)}")
    print(f"Answer model: {answer_model} | Judge: {judge_model} | top_k: {top_k}")
    print(f"Output: {output_file}")
    print("=" * 60)

    results = []
    cat_scores = {}
    total_start = time.time()
    done_count = 0

    for i, q_item in enumerate(qa_list):
        category = q_item.get("category", -1)
        if category == 5:
            continue

        question = q_item["question"]
        gold_answer = str(q_item["answer"])
        evidence = q_item.get("evidence", [])
        adversarial_answer = q_item.get("adversarial_answer", "")

        # --- 1. Search ---
        search_start = time.time()
        try:
            res_a = mem0_client.search(question, filters={"user_id": speaker_a_id}, top_k=top_k)
            mem_a = res_a.get("results", res_a) if isinstance(res_a, dict) else res_a
        except Exception as e:
            mem_a = []
            print(f"  [{i}] Search A error: {e}")
        search_a_time = time.time() - search_start

        search_start2 = time.time()
        try:
            res_b = mem0_client.search(question, filters={"user_id": speaker_b_id}, top_k=top_k)
            mem_b = res_b.get("results", res_b) if isinstance(res_b, dict) else res_b
        except Exception as e:
            mem_b = []
            print(f"  [{i}] Search B error: {e}")
        search_b_time = time.time() - search_start2

        speaker_1_memories = [
            {"memory": m["memory"], "timestamp": m.get("metadata", {}).get("timestamp", "?"),
             "score": round(m.get("score", 0), 4)}
            for m in mem_a
        ]
        speaker_2_memories = [
            {"memory": m["memory"], "timestamp": m.get("metadata", {}).get("timestamp", "?"),
             "score": round(m.get("score", 0), 4)}
            for m in mem_b
        ]

        search_1_formatted = [f"{m['timestamp']}: {m['memory']}" for m in speaker_1_memories]
        search_2_formatted = [f"{m['timestamp']}: {m['memory']}" for m in speaker_2_memories]

        # --- 2. Answer ---
        tmpl = Template(ANSWER_PROMPT)
        prompt = tmpl.render(
            speaker_1_user_id=speaker_a,
            speaker_2_user_id=speaker_b,
            speaker_1_memories=json.dumps(search_1_formatted, indent=4),
            speaker_2_memories=json.dumps(search_2_formatted, indent=4),
            question=question,
        )

        answer_start = time.time()
        try:
            resp = openai_client.chat.completions.create(
                model=answer_model,
                messages=[{"role": "system", "content": prompt}],
            )
            pred_answer = resp.choices[0].message.content
            answer_tokens = resp.usage.total_tokens if resp.usage else 0
        except Exception as e:
            pred_answer = f"[ERROR: {e}]"
            answer_tokens = 0
            print(f"  [{i}] Answer error: {e}")
        answer_time = time.time() - answer_start

        # --- 3. Evaluate ---
        bleu = round(calc_bleu1(pred_answer, gold_answer), 4)
        f1 = round(calc_f1(pred_answer, gold_answer), 4)
        score, judge_raw = llm_judge(question, gold_answer, pred_answer, openai_client, judge_model)

        result = {
            "question": question,
            "answer": gold_answer,
            "response": pred_answer,
            "category": category,
            "evidence": evidence,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": round(search_a_time, 3),
            "speaker_2_memory_time": round(search_b_time, 3),
            "response_time": round(answer_time, 3),
            "answer_tokens": answer_tokens,
            "bleu_score": bleu,
            "f1_score": f1,
            "llm_score": score,
            "llm_judge_raw": judge_raw,
        }
        results.append(result)

        if category not in cat_scores:
            cat_scores[category] = {"correct": 0, "total": 0, "bleu_sum": 0, "f1_sum": 0}
        cat_scores[category]["total"] += 1
        cat_scores[category]["correct"] += score
        cat_scores[category]["bleu_sum"] += bleu
        cat_scores[category]["f1_sum"] += f1

        done_count += 1
        overall_acc = sum(r["llm_score"] for r in results) / done_count * 100
        mark = "O" if score else "X"
        print(f"  [{done_count}/{len(evaluated)}] cat{category} {mark} | "
              f"pred: {pred_answer[:50]}... | gold: {gold_answer[:35]} | "
              f"mem:{len(mem_a)}+{len(mem_b)} | acc:{overall_acc:.1f}%")

        with open(output_file, "w") as f:
            json.dump({"conv_idx": conv_idx, "results": results}, f, indent=2)

    # --- Final summary ---
    elapsed = time.time() - total_start
    total_q = len(results)
    total_correct = sum(r["llm_score"] for r in results)
    avg_bleu = sum(r["bleu_score"] for r in results) / total_q if total_q else 0
    avg_f1 = sum(r["f1_score"] for r in results) / total_q if total_q else 0
    avg_search = sum(r["speaker_1_memory_time"] + r["speaker_2_memory_time"] for r in results) / total_q if total_q else 0
    avg_answer = sum(r["response_time"] for r in results) / total_q if total_q else 0
    avg_tokens = sum(r["answer_tokens"] for r in results) / total_q if total_q else 0

    cat_names = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop", 4: "Open-domain"}
    per_cat = {}
    for cat in sorted(cat_scores.keys()):
        s = cat_scores[cat]
        per_cat[cat] = {
            "type": cat_names.get(cat, "?"),
            "llm_judge": f"{s['correct']}/{s['total']}",
            "llm_judge_pct": round(s["correct"] / s["total"] * 100, 1) if s["total"] else 0,
            "bleu1": round(s["bleu_sum"] / s["total"], 4) if s["total"] else 0,
            "f1": round(s["f1_sum"] / s["total"], 4) if s["total"] else 0,
            "count": s["total"],
        }

    summary = {
        "conv_idx": conv_idx,
        "speakers": f"{speaker_a} <-> {speaker_b}",
        "answer_model": answer_model,
        "judge_model": judge_model,
        "top_k": top_k,
        "total_questions": total_q,
        "llm_judge_accuracy": round(total_correct / total_q * 100, 1) if total_q else 0,
        "llm_judge_correct": total_correct,
        "bleu1_avg": round(avg_bleu, 4),
        "f1_avg": round(avg_f1, 4),
        "avg_search_time_s": round(avg_search, 3),
        "avg_answer_time_s": round(avg_answer, 3),
        "avg_answer_tokens": round(avg_tokens, 1),
        "total_time_s": round(elapsed, 1),
        "per_category": per_cat,
    }

    full_output = {"summary": summary, "results": results}
    with open(output_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"CONV {conv_idx} EVALUATION RESULTS")
    print("=" * 60)
    print(f"Speakers:            {speaker_a} <-> {speaker_b}")
    print(f"Questions evaluated:  {total_q}")
    print(f"LLM-Judge Accuracy:  {total_correct}/{total_q} = {summary['llm_judge_accuracy']}%")
    print(f"BLEU-1 (avg):        {summary['bleu1_avg']}")
    print(f"F1 (avg):            {summary['f1_avg']}")
    print(f"Avg search time:     {summary['avg_search_time_s']}s")
    print(f"Avg answer time:     {summary['avg_answer_time_s']}s")
    print(f"Avg answer tokens:   {summary['avg_answer_tokens']}")
    print(f"Total time:          {int(elapsed//60)}m {int(elapsed%60)}s")

    print(f"\nPer-Category:")
    print(f"{'Cat':<5} {'Type':<12} {'LLM-Judge':<18} {'BLEU-1':<10} {'F1':<10} {'Count'}")
    print("-" * 65)
    for cat in sorted(per_cat.keys()):
        c = per_cat[cat]
        print(f"  {cat:<3} {c['type']:<12} {c['llm_judge']} = {c['llm_judge_pct']}%"
              f"     {c['bleu1']:<10} {c['f1']:<10} {c['count']}")
    print("=" * 60)
    print(f"Full results saved to: {output_file}")


if __name__ == "__main__":
    main()
