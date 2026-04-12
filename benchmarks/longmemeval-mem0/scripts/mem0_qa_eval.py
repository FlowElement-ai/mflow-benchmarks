#!/usr/bin/env python3
"""
Mem0 LongMemEval QA 评估脚本
完全对齐 mflow_qa_eval.py 的评估流程、提示词、指标、输出格式

关键对齐点:
  - ANSWER_PROMPT / JUDGE_PROMPT 与 mflow 完全一致
  - 答题模型 gpt-5-mini、评判模型 gpt-4o-mini
  - 指标计算 BLEU-1、F1、LLM-Judge 实现完全一致
  - 输出 JSON 结构 {summary, results} 与 mflow 兼容
  - 可直接用 analyze_results.py 进行 MFlow vs Mem0 对比
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Prompts — 与 mflow_qa_eval.py 完全一致（逐字复制）
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
# 数据加载 — 与 mflow 一致的数据路径搜索
# ---------------------------------------------------------------------------

def find_data_file() -> Path:
    possible_paths = [
        Path(os.environ.get('BENCHMARK_WORKSPACE', '')) / "graphiti-main/tests/evals/data/longmemeval_data/longmemeval_oracle.json",
        PROJECT_ROOT / "data" / "longmemeval_oracle.json",
        SCRIPT_DIR / "data" / "longmemeval_oracle.json",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"LongMemEval 数据文件未找到。请将数据文件放置于:\n"
        f"  - {possible_paths[1]}"
    )


DATA_PATH = find_data_file()


def load_questions(max_questions: int = 50) -> List[Dict]:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions[:max_questions]


# ---------------------------------------------------------------------------
# 指标计算 — 与 mflow_qa_eval.py 完全一致（逐行复制）
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

    if len(pred_tokens) == 0:
        return 0.0

    smooth = SmoothingFunction().method1
    try:
        return sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    except (ValueError, ZeroDivisionError):
        return 0.0


def calculate_f1(prediction: str, reference: str) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    def tokenize(text: str) -> set:
        text = str(text).lower()
        text = text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
        return set(text.split())

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = pred_tokens & ref_tokens
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_llm_judge(question: str, gold: str, generated: str, client: OpenAI, model: str) -> int:
    """LLM 判断答案正确性 — 与 mflow 完全一致"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold_answer=gold, generated_answer=generated
            )}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        return 1 if result.get("label", "").upper() == "CORRECT" else 0
    except Exception as e:
        print(f"  LLM Judge 错误: {e}")
        return 0


# ---------------------------------------------------------------------------
# Mem0 记忆检索 — 带重试和多格式兼容
# ---------------------------------------------------------------------------

def _parse_search_response(results, top_k: int) -> list:
    """
    解析 mem0 search() 返回值，兼容多种 SDK 版本的响应格式:
      - dict: {"memories": [...]}  或 {"results": [...]}
      - list: [...]  直接返回列表
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


def retrieve_memories(mem0_client, question: str, question_id: str, top_k: int = 10) -> str:
    """
    使用 Mem0 语义检索记忆，带重试逻辑
    对应 mflow_qa_eval.py 的 retrieve_memories()
    - user_id = lme_{question_id}（对应 mflow 的 dataset_name）
    - top_k = 10（与 mflow config.yaml 一致）
    """
    user_id = f"lme_{question_id}"

    for attempt in range(3):
        try:
            results = mem0_client.search(
                question,
                filters={"user_id": user_id},
                top_k=top_k,
            )
            parsed = _parse_search_response(results, top_k)
            return "\n\n".join(parsed) if parsed else ""
        except TypeError:
            try:
                results = mem0_client.search(
                    question,
                    filters={"user_id": user_id},
                )
                parsed = _parse_search_response(results, top_k)
                return "\n\n".join(parsed) if parsed else ""
            except Exception as e2:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"  Mem0 检索错误 (降级后仍失败): {e2}")
                return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  Mem0 检索错误 (尝试 {attempt+1}/3): {e}")
            return ""

    return ""


# ---------------------------------------------------------------------------
# 答案生成 — 与 mflow 完全一致
# ---------------------------------------------------------------------------

def generate_answer(memories: str, question: str, client: OpenAI, model: str) -> str:
    if not memories:
        return "No relevant information found"

    try:
        _is_reasoning = model.startswith("gpt-5") or model.startswith("o")
        params = dict(
            model=model,
            messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                memories=memories, question=question
            )}],
        )
        if _is_reasoning:
            params["max_completion_tokens"] = 2048
        else:
            params["temperature"] = 0.0
            params["max_tokens"] = 50
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  生成答案错误: {e}")
        return "Error generating answer"


# ---------------------------------------------------------------------------
# 单题评估
# ---------------------------------------------------------------------------

def evaluate_single(
    question: Dict,
    question_idx: int,
    total: int,
    mem0_client,
    openai_client: OpenAI,
    answer_model: str,
    judge_model: str,
    top_k: int,
) -> Dict:
    """评估单个问题 — 流程与 mflow_qa_eval.py 的 evaluate_single() 完全一致"""
    question_id = question['question_id']
    q_text = question['question']
    gold_answer = str(question['answer'])

    print(f"[{question_idx+1}/{total}] Mem0: {question_id}")

    start_time = time.time()

    # 1. 检索
    retrieval_start = time.time()
    memories = retrieve_memories(mem0_client, q_text, question_id, top_k=top_k)
    retrieval_ms = (time.time() - retrieval_start) * 1000

    # 2. 生成答案
    generation_start = time.time()
    generated = generate_answer(memories, q_text, openai_client, answer_model)
    generation_ms = (time.time() - generation_start) * 1000

    # 3. 计算指标（与 mflow 完全一致）
    bleu = calculate_bleu1(generated, gold_answer)
    f1 = calculate_f1(generated, gold_answer)
    llm_score = evaluate_llm_judge(q_text, gold_answer, generated, openai_client, judge_model)

    total_ms = (time.time() - start_time) * 1000

    mem_count = len(memories.split('\n\n')) if memories else 0
    print(f"  答案: {generated[:60]}{'...' if len(generated)>60 else ''}")
    print(f"  Gold: {gold_answer} | BLEU: {bleu:.3f}, F1: {f1:.3f}, LLM: {'OK' if llm_score else 'WRONG'} | {mem_count} memories, {retrieval_ms:.0f}ms")

    return {
        'question_id': question_id,
        'question': q_text,
        'question_type': question.get('question_type', ''),
        'gold_answer': gold_answer,
        'generated_answer': generated,
        'memories_retrieved': memories[:500] if memories else "",
        'memories_count': mem_count,
        'bleu_score': round(bleu, 4),
        'f1_score': round(f1, 4),
        'llm_score': llm_score,
        'retrieval_ms': round(retrieval_ms, 2),
        'generation_ms': round(generation_ms, 2),
        'total_ms': round(total_ms, 2),
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Mem0 LongMemEval QA 评估')
    parser.add_argument('--max-questions', type=int, default=50, help='评估问题数 (默认: 50)')
    parser.add_argument('--start-from', type=int, default=0, help='从第 N 个问题开始')
    parser.add_argument('--answer-model', type=str, default='gpt-5-mini', help='答题模型 (默认: gpt-5-mini)')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini', help='评判模型 (默认: gpt-4o-mini)')
    parser.add_argument('--top-k', type=int, default=10, help='检索记忆数量 (默认: 10)')
    parser.add_argument('--output-dir', type=str, default='results', help='输出目录')
    parser.add_argument('--merge-results', type=str, default='', help='合并之前的结果文件路径 (断点续传)')
    parser.add_argument('--api-delay', type=float, default=0.5, help='API 调用间延迟秒数 (默认: 0.5)')
    parser.add_argument('--mem0-api-key', type=str, default='', help='Mem0 API Key')
    parser.add_argument('--openai-api-key', type=str, default='', help='OpenAI API Key')
    args = parser.parse_args()

    mem0_key = args.mem0_api_key or MEM0_API_KEY
    openai_key = args.openai_api_key or OPENAI_API_KEY

    if not mem0_key:
        print("ERROR: 必须提供 Mem0 API Key。")
        print("  方式1: --mem0-api-key m0-xxx")
        print("  方式2: export MEM0_API_KEY=m0-xxx")
        sys.exit(1)
    if not openai_key:
        print("ERROR: 必须提供 OpenAI API Key。")
        print("  方式1: --openai-api-key sk-xxx")
        print("  方式2: export OPENAI_API_KEY=sk-xxx")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = openai_key

    from mem0 import MemoryClient
    mem0_client = MemoryClient(api_key=mem0_key)
    openai_client = OpenAI(api_key=openai_key)

    # 验证连接
    print(f"\n{'='*60}")
    print(f"Mem0 LongMemEval 评估")
    print(f"{'='*60}")
    print(f"答题模型: {args.answer_model}")
    print(f"评判模型: {args.judge_model}")
    print(f"检索 top_k: {args.top_k}")
    print(f"API 延迟: {args.api_delay}s")

    # 快速验证 mem0 连接
    try:
        test_result = mem0_client.search("test", filters={"user_id": "__connectivity_test__"})
        print(f"Mem0 连接: OK")
    except Exception as e:
        print(f"Mem0 连接测试: {e}")
        print(f"(将继续运行，但如果持续失败请检查 API Key)")

    print(f"{'='*60}")

    all_questions = load_questions(args.max_questions)
    questions = all_questions[args.start_from:]
    print(f"加载 {len(all_questions)} 个问题, 从第 {args.start_from} 题开始, 评估 {len(questions)} 题")

    # 断点续传
    results = []
    evaluated_ids = set()
    if args.merge_results:
        try:
            with open(args.merge_results, 'r') as f:
                prev = json.load(f)
            results = prev.get('results', [])
            evaluated_ids = {r['question_id'] for r in results}
            print(f"已加载之前的 {len(results)} 条结果")
        except Exception as e:
            print(f"加载之前结果失败: {e}")

    start_time = time.time()

    for idx, q in enumerate(questions):
        qid = q.get('question_id', '')
        if qid in evaluated_ids:
            print(f"[{args.start_from + idx + 1}/{len(all_questions)}] 跳过已评估: {qid}")
            continue

        result = evaluate_single(
            q, args.start_from + idx, len(all_questions),
            mem0_client, openai_client,
            args.answer_model, args.judge_model,
            args.top_k,
        )
        results.append(result)

        # 每 10 题自动保存一次（防止中途崩溃丢失结果）
        if len(results) % 10 == 0:
            _save_intermediate(results, args, start_time)

        if args.api_delay > 0:
            time.sleep(args.api_delay)

    total_time = time.time() - start_time

    # 计算汇总统计 — 输出格式与 mflow 完全一致
    n = len(results) if results else 1
    summary = {
        'engine': 'mem0',
        'total_questions': len(results),
        'answer_model': args.answer_model,
        'judge_model': args.judge_model,
        'top_k': args.top_k,
        'avg_bleu': round(sum(r['bleu_score'] for r in results) / n, 4) if results else 0,
        'avg_f1': round(sum(r['f1_score'] for r in results) / n, 4) if results else 0,
        'llm_accuracy': round(sum(r['llm_score'] for r in results) / n, 4) if results else 0,
        'avg_retrieval_ms': round(sum(r['retrieval_ms'] for r in results) / n, 2) if results else 0,
        'avg_generation_ms': round(sum(r['generation_ms'] for r in results) / n, 2) if results else 0,
        'total_time_seconds': round(total_time, 2),
    }

    # 按 question_type 统计
    type_stats = {}
    for r in results:
        qt = r.get('question_type', 'unknown')
        if qt not in type_stats:
            type_stats[qt] = {'correct': 0, 'total': 0}
        type_stats[qt]['total'] += 1
        type_stats[qt]['correct'] += r['llm_score']

    for qt in type_stats:
        t = type_stats[qt]
        t['accuracy'] = round(t['correct'] / t['total'], 4) if t['total'] > 0 else 0
    summary['per_type_accuracy'] = type_stats

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Mem0 评估完成")
    print(f"{'='*60}")
    correct = sum(r['llm_score'] for r in results)
    print(f"LLM-Judge Accuracy: {summary['llm_accuracy']:.4f} ({correct}/{len(results)})")
    print(f"BLEU-1: {summary['avg_bleu']:.4f}")
    print(f"F1: {summary['avg_f1']:.4f}")
    print(f"平均检索延迟: {summary['avg_retrieval_ms']:.2f}ms")
    print(f"平均生成延迟: {summary['avg_generation_ms']:.2f}ms")
    print(f"总耗时: {summary['total_time_seconds']:.2f}s")

    if type_stats:
        print(f"\n按题型分类:")
        for qt, stats in sorted(type_stats.items()):
            print(f"  {qt}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.1f}%)")

    zero_mem = sum(1 for r in results if r['memories_count'] == 0)
    if zero_mem > 0:
        print(f"\n[WARN] {zero_mem}/{len(results)} 个问题未检索到任何记忆")

    # 保存结果
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mem0_eval_results.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


def _save_intermediate(results: list, args, start_time: float):
    """中间自动保存"""
    try:
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "mem0_eval_results_partial.json"
        n = len(results) if results else 1
        summary = {
            'engine': 'mem0',
            'total_questions': len(results),
            'llm_accuracy': round(sum(r['llm_score'] for r in results) / n, 4),
            'total_time_seconds': round(time.time() - start_time, 2),
            'status': 'partial',
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'results': results}, f, indent=2, ensure_ascii=False)
        print(f"  [自动保存: {len(results)} 条结果 -> {path.name}]")
    except Exception:
        pass


if __name__ == '__main__':
    main()
