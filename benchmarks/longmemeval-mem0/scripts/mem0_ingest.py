#!/usr/bin/env python3
"""
Mem0 LongMemEval 入库脚本
按问题隔离入库，每个问题的数据使用独立的 user_id
完全对齐 mflow_ingest.py 的流程和参数

关键设计决策:
  1. mem0 add() 始终异步处理 (async_mode=False 已废弃)，
     因此入库后需等待 mem0 后台处理完毕再进行评估
  2. user_id = lme_{question_id} — 等价于 mflow 的 dataset_name
  3. timestamp 传入原始会话日期 — 避免 mem0 使用当前日期的已知问题 (Issue #3944)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")


def find_data_file() -> Path:
    """查找 LongMemEval 数据文件，尝试多个可能的位置（与 mflow 一致）"""
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


def load_questions() -> list:
    """加载 LongMemEval 问题数据"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"  已加载 {len(questions)} 个问题")
    return questions


def parse_date_to_unix(date_str: str) -> int:
    """
    解析 LongMemEval 日期格式: '2023/04/10 (Mon) 17:50'
    返回 Unix 时间戳（秒），与 mflow 的 parse_date() 对齐
    """
    dt = datetime.strptime(
        date_str + ' UTC',
        '%Y/%m/%d (%a) %H:%M UTC'
    ).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def add_with_retry(client, messages, user_id, timestamp, metadata, max_retries=3, retry_delay=5.0):
    """
    带重试的 add() 调用
    注意: mem0 add() 始终异步处理 (async_mode=False 已废弃)，
    入库后需等待 mem0 后台完成处理才能搜索到记忆
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            result = client.add(
                messages,
                user_id=user_id,
                timestamp=timestamp,
                metadata=metadata,
            )
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"    add() 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                print(f"    等待 {wait:.0f}s 后重试...")
                time.sleep(wait)
            else:
                raise last_error


def ingest_question(client, question: dict, question_idx: int, total: int, session_delay: float) -> dict:
    """
    入库单个问题的数据
    对齐 mflow_ingest.py 的 ingest_question():
    - 每个问题一个独立的 user_id（对应 mflow 的 dataset_name）
    - 按会话逐个入库（对应 mflow 的 per-session add+memorize）
    - 传入时间戳（对应 mflow 的 created_at）
    """
    question_id = question.get('question_id', f'q_{question_idx}')
    user_id = f"lme_{question_id}"

    sessions = question.get('haystack_sessions', [])
    dates = question.get('haystack_dates', [])

    if len(sessions) != len(dates):
        print(f"  [WARN] 问题 {question_id}: sessions ({len(sessions)}) 和 dates ({len(dates)}) 数量不匹配")
        return {'question_id': question_id, 'status': 'error', 'error': 'length_mismatch'}

    total_messages = sum(len(s) for s in sessions)
    start_time = time.time()

    print(f"[{question_idx+1}/{total}] 入库问题 {question_id}: {len(sessions)} 个会话, {total_messages} 条消息")

    ingested_sessions = 0
    try:
        for session_idx, session in enumerate(sessions):
            date_str = dates[session_idx]
            unix_ts = parse_date_to_unix(date_str)

            # 只提取 role 和 content，排除 has_answer 等额外字段
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session
            ]

            add_with_retry(
                client,
                messages,
                user_id=user_id,
                timestamp=unix_ts,
                metadata={"session_date": date_str, "session_idx": session_idx},
            )
            ingested_sessions += 1

            if session_delay > 0 and session_idx < len(sessions) - 1:
                time.sleep(session_delay)

        elapsed = time.time() - start_time
        print(f"  OK ({elapsed:.1f}s, {ingested_sessions}/{len(sessions)} sessions)")

        return {
            'question_id': question_id,
            'status': 'success',
            'sessions': len(sessions),
            'ingested_sessions': ingested_sessions,
            'messages': total_messages,
            'elapsed_seconds': round(elapsed, 2)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  FAIL ({ingested_sessions}/{len(sessions)} sessions): {e}")
        return {
            'question_id': question_id,
            'status': 'error',
            'error': str(e),
            'ingested_sessions': ingested_sessions,
            'elapsed_seconds': round(elapsed, 2)
        }


def main():
    parser = argparse.ArgumentParser(description='Mem0 LongMemEval 入库脚本')
    parser.add_argument('--max-questions', type=int, default=50, help='最大处理问题数 (默认: 50)')
    parser.add_argument('--start-from', type=int, default=0, help='从第 N 个问题开始 (默认: 0)')
    parser.add_argument('--session-delay', type=float, default=2.0,
                        help='会话间延迟秒数，等待 mem0 异步处理 (默认: 2.0)')
    parser.add_argument('--api-key', type=str, default='', help='Mem0 API Key (也可通过 MEM0_API_KEY 环境变量设置)')
    parser.add_argument('--clean', action='store_true', help='入库前清除对应 user_id 的旧记忆')
    args = parser.parse_args()

    api_key = args.api_key or MEM0_API_KEY
    if not api_key:
        print("ERROR: 必须提供 Mem0 API Key。")
        print("  方式1: --api-key m0-xxx")
        print("  方式2: export MEM0_API_KEY=m0-xxx")
        sys.exit(1)

    from mem0 import MemoryClient
    client = MemoryClient(api_key=api_key)

    print("=" * 60)
    print("Mem0 LongMemEval 入库脚本")
    print("=" * 60)
    print(f"数据路径: {DATA_PATH}")
    print(f"处理范围: 问题 {args.start_from} - {args.start_from + args.max_questions - 1}")
    print(f"会话间延迟: {args.session_delay}s")
    print(f"清除旧记忆: {args.clean}")
    print("=" * 60)

    questions = load_questions()

    end_idx = min(args.start_from + args.max_questions, len(questions))
    questions_to_process = questions[args.start_from:end_idx]

    print(f"\n开始入库 {len(questions_to_process)} 个问题...\n")

    # 清除旧记忆
    if args.clean:
        print("正在清除旧记忆...")
        for idx, question in enumerate(questions_to_process):
            question_id = question.get('question_id', f'q_{args.start_from + idx}')
            user_id = f"lme_{question_id}"
            try:
                client.delete_all(user_id=user_id)
                print(f"  [{idx+1}/{len(questions_to_process)}] 已清除 {user_id}")
            except Exception as e:
                print(f"  [{idx+1}/{len(questions_to_process)}] 清除 {user_id} 失败: {e}")
            time.sleep(0.3)
        print("清除完成\n")

    results = []
    total_start = time.time()

    for idx, question in enumerate(questions_to_process):
        result = ingest_question(
            client, question, args.start_from + idx, end_idx, args.session_delay
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    total_messages = sum(r.get('messages', 0) for r in results if r['status'] == 'success')
    total_sessions = sum(r.get('ingested_sessions', 0) for r in results)

    print("\n" + "=" * 60)
    print("入库完成")
    print("=" * 60)
    print(f"成功: {success_count}, 失败: {error_count}")
    print(f"总会话数: {total_sessions}")
    print(f"总消息数: {total_messages}")
    print(f"总耗时: {total_elapsed:.1f}s")
    if len(questions_to_process) > 0:
        print(f"平均: {total_elapsed/len(questions_to_process):.1f}s/问题")

    # 保存结果
    result_file = PROJECT_ROOT / "results" / f"mem0_ingest_results_{args.start_from}_{end_idx}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'engine': 'mem0',
            'start_idx': args.start_from,
            'end_idx': end_idx,
            'total_elapsed': round(total_elapsed, 2),
            'success_count': success_count,
            'error_count': error_count,
            'total_sessions': total_sessions,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {result_file}")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
