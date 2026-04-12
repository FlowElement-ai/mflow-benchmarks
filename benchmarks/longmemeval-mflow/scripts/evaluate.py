#!/usr/bin/env python3
"""
MFlow LongMemEval QA 评估脚本
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def find_mflow_root() -> Path:
    """Find MFlow installation root directory"""
    if os.environ.get('MFLOW_ROOT'):
        return Path(os.environ['MFLOW_ROOT'])
    in_tree = SCRIPT_DIR.parent.parent
    if (in_tree / 'm_flow').is_dir():
        return in_tree
    raise RuntimeError(
        "MFlow root not found. Set MFLOW_ROOT environment variable:\n"
        "  export MFLOW_ROOT=/path/to/mflow"
    )


MFLOW_ROOT = find_mflow_root()
os.chdir(MFLOW_ROOT)

env_path = MFLOW_ROOT / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value.startswith('"') and '"' in value[1:]:
                    value = value[1:value.index('"', 1)]
                elif value.startswith("'") and "'" in value[1:]:
                    value = value[1:value.index("'", 1)]
                else:
                    if '#' in value:
                        value = value[:value.index('#')].strip()
                os.environ[key] = value

# 现在添加路径并导入
sys.path.insert(0, str(MFLOW_ROOT))

import asyncio
import argparse
import json
import time
from typing import Dict, List

from openai import OpenAI
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# 下载 NLTK 数据
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

def find_data_file() -> Path:
    """查找 LongMemEval 数据文件，尝试多个可能的位置"""
    possible_paths = [
        Path(os.environ.get('BENCHMARK_WORKSPACE', '')) / "graphiti-main/tests/evals/data/longmemeval_data/longmemeval_oracle.json",
        PROJECT_ROOT / "data" / "longmemeval_oracle.json",
        SCRIPT_DIR / "data" / "longmemeval_oracle.json",
        MFLOW_ROOT.parent / "graphiti-main/tests/evals/data/longmemeval_data/longmemeval_oracle.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"LongMemEval 数据文件未找到。请设置环境变量 BENCHMARK_WORKSPACE 或将数据文件放置于:\n"
        f"  - {possible_paths[1]}\n"
        f"  - {possible_paths[3]}"
    )


DATA_PATH = find_data_file()

# Prompts
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


def load_questions(max_questions: int = 50) -> List[Dict]:
    """加载 LongMemEval 问题"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions[:max_questions]


def calculate_bleu1(prediction: str, reference: str) -> float:
    """计算 BLEU-1"""
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
    """计算 F1"""
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
    """LLM 判断答案正确性"""
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


async def retrieve_memories(question: str, question_id: str, top_k: int = 10) -> str:
    """使用 MFlow 检索记忆"""
    from m_flow.retrieval.episodic_retriever import EpisodicRetriever
    from m_flow.retrieval.episodic import EpisodicConfig
    from m_flow.context_global_variables import (
        backend_access_control_enabled,
        set_db_context,
    )
    from m_flow.data.methods import get_datasets_by_name
    from m_flow.auth.methods.get_seed_user import get_seed_user
    
    dataset_name = f"lme_{question_id}"
    
    try:
        # 设置 dataset 上下文（多用户模式需要）
        if backend_access_control_enabled():
            default_user = await get_seed_user()
            datasets = await get_datasets_by_name(dataset_name, default_user.id)
            if datasets:
                dataset = datasets[0]
                await set_db_context(dataset.id, dataset.owner_id)
            else:
                print(f"  Warning: Dataset '{dataset_name}' not found")
                return ""
        
        config = EpisodicConfig(
            top_k=top_k,
            wide_search_top_k=top_k * 3,
            display_mode="summary",
        )
        retriever = EpisodicRetriever(config=config)
        edges = await retriever.get_triplets(question)
        
        if not edges:
            return ""
        
        # 提取记忆文本（参考 search_aligned.py 的方式）
        memories = []
        seen_episodes = set()
        
        for edge in edges:
            # 检查两个节点
            for node in (getattr(edge, 'node1', None), getattr(edge, 'node2', None)):
                if node is None:
                    continue
                attrs = getattr(node, 'attributes', {}) or {}
                if attrs.get("type") == "Episode":
                    node_id = str(getattr(node, 'id', ''))
                    if node_id in seen_episodes:
                        continue
                    seen_episodes.add(node_id)
                    
                    # 获取 Episode 的 summary
                    summary = attrs.get("summary", "")
                    if summary:
                        memories.append(summary)
            
            # 也尝试获取 edge 的名称
            edge_name = getattr(edge, 'name', None)
            if edge_name and edge_name not in memories:
                memories.append(edge_name)
        
        return "\n\n".join(memories[:top_k]) if memories else ""
    except Exception as e:
        print(f"  MFlow 检索错误: {e}")
        return ""


async def generate_answer(memories: str, question: str, client: OpenAI, model: str) -> str:
    """生成答案"""
    if not memories:
        return "No relevant information found"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                memories=memories, question=question
            )}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  生成答案错误: {e}")
        return "Error generating answer"


async def evaluate_single(
    question: Dict,
    question_idx: int,
    total: int,
    client: OpenAI,
    answer_model: str,
    judge_model: str,
) -> Dict:
    """评估单个问题"""
    question_id = question['question_id']
    q_text = question['question']
    gold_answer = str(question['answer'])
    
    print(f"[{question_idx+1}/{total}] MFlow: {question_id}")
    
    start_time = time.time()
    
    # 1. 检索
    retrieval_start = time.time()
    memories = await retrieve_memories(q_text, question_id)
    retrieval_ms = (time.time() - retrieval_start) * 1000
    
    # 2. 生成答案
    generation_start = time.time()
    generated = await generate_answer(memories, q_text, client, answer_model)
    generation_ms = (time.time() - generation_start) * 1000
    
    # 3. 计算指标
    bleu = calculate_bleu1(generated, gold_answer)
    f1 = calculate_f1(generated, gold_answer)
    llm_score = evaluate_llm_judge(q_text, gold_answer, generated, client, judge_model)
    
    total_ms = (time.time() - start_time) * 1000
    
    print(f"  答案: {generated[:50]}... | BLEU: {bleu:.3f}, F1: {f1:.3f}, LLM: {llm_score}")
    
    return {
        'question_id': question_id,
        'question': q_text,
        'gold_answer': gold_answer,
        'generated_answer': generated,
        'memories_count': len(memories.split('\n\n')) if memories else 0,
        'bleu_score': round(bleu, 4),
        'f1_score': round(f1, 4),
        'llm_score': llm_score,
        'retrieval_ms': round(retrieval_ms, 2),
        'generation_ms': round(generation_ms, 2),
        'total_ms': round(total_ms, 2),
    }


async def main():
    parser = argparse.ArgumentParser(description='MFlow LongMemEval QA 评估')
    parser.add_argument('--max-questions', type=int, default=50, help='评估问题数')
    parser.add_argument('--start-from', type=int, default=0, help='从第 N 个问题开始')
    parser.add_argument('--answer-model', type=str, default='gpt-4.1-mini', help='答题模型')
    parser.add_argument('--judge-model', type=str, default='gpt-4.1-mini', help='评判模型')
    parser.add_argument('--output-dir', type=str, default='results', help='输出目录')
    parser.add_argument('--merge-results', type=str, default='', help='合并之前的结果文件路径')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"MFlow LongMemEval 评估")
    print(f"{'='*60}")
    print(f"EMBEDDING_DIMENSIONS: {os.environ.get('EMBEDDING_DIMENSIONS', 'NOT SET')}")
    print(f"{'='*60}")
    
    # 加载问题
    all_questions = load_questions(args.max_questions)
    questions = all_questions[args.start_from:]
    print(f"加载 {len(all_questions)} 个问题, 从第 {args.start_from} 题开始, 评估 {len(questions)} 题")
    
    # OpenAI 客户端
    client = OpenAI()
    
    # 加载之前的结果（断点续传）
    results = []
    if args.merge_results:
        try:
            with open(args.merge_results, 'r') as f:
                prev = json.load(f)
            results = prev.get('results', [])
            print(f"已加载之前的 {len(results)} 条结果")
        except Exception as e:
            print(f"加载之前结果失败: {e}")
    
    start_time = time.time()
    
    for idx, q in enumerate(questions):
        result = await evaluate_single(
            q, args.start_from + idx, len(all_questions),
            client, args.answer_model, args.judge_model
        )
        results.append(result)
        
        # 避免 API 限流
        await asyncio.sleep(0.5)
    
    total_time = time.time() - start_time
    
    # 计算汇总统计
    n = len(results) if results else 1  # 防止除零
    summary = {
        'engine': 'mflow',
        'total_questions': len(results),
        'answer_model': args.answer_model,
        'judge_model': args.judge_model,
        'avg_bleu': round(sum(r['bleu_score'] for r in results) / n, 4) if results else 0,
        'avg_f1': round(sum(r['f1_score'] for r in results) / n, 4) if results else 0,
        'llm_accuracy': round(sum(r['llm_score'] for r in results) / n, 4) if results else 0,
        'avg_retrieval_ms': round(sum(r['retrieval_ms'] for r in results) / n, 2) if results else 0,
        'avg_generation_ms': round(sum(r['generation_ms'] for r in results) / n, 2) if results else 0,
        'total_time_seconds': round(total_time, 2),
    }
    
    print(f"\n{'='*60}")
    print(f"MFlow 评估完成")
    print(f"{'='*60}")
    print(f"BLEU-1: {summary['avg_bleu']:.4f}")
    print(f"F1: {summary['avg_f1']:.4f}")
    print(f"LLM-Judge Accuracy: {summary['llm_accuracy']:.4f}")
    print(f"平均检索延迟: {summary['avg_retrieval_ms']:.2f}ms")
    print(f"平均生成延迟: {summary['avg_generation_ms']:.2f}ms")
    print(f"总耗时: {summary['total_time_seconds']:.2f}s")
    
    # 保存结果
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mflow_eval_results.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
