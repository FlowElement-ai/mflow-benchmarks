#!/usr/bin/env python3
"""
MFlow LongMemEval 入库脚本
按问题隔离入库，每个问题的数据使用独立的 dataset_name
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
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
sys.path.insert(0, str(MFLOW_ROOT))

from dotenv import load_dotenv
load_dotenv(MFLOW_ROOT / '.env')

from m_flow import ContentType
import m_flow


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


def load_questions() -> list:
    """加载 LongMemEval 问题数据"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"✓ 已加载 {len(questions)} 个问题")
    return questions


def parse_date(date_str: str) -> datetime:
    """
    解析 LongMemEval 日期格式: '2023/04/10 (Mon) 17:50'
    返回 UTC 时区的 datetime 对象
    """
    return datetime.strptime(
        date_str + ' UTC', 
        '%Y/%m/%d (%a) %H:%M UTC'
    ).replace(tzinfo=timezone.utc)


def format_session(session: list, date_str: str) -> str:
    """
    将 session 消息列表格式化为文本
    包含时间戳信息，便于时间推理
    """
    lines = [f"[Session Date: {date_str}]"]
    for msg in session:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


async def ingest_question(question: dict, question_idx: int, total: int) -> dict:
    """入库单个问题的数据"""
    question_id = question.get('question_id', f'q_{question_idx}')
    dataset_name = f"lme_{question_id}"
    
    sessions = question.get('haystack_sessions', [])
    dates = question.get('haystack_dates', [])
    
    if len(sessions) != len(dates):
        print(f"⚠️ 问题 {question_id}: sessions ({len(sessions)}) 和 dates ({len(dates)}) 数量不匹配")
        return {'question_id': question_id, 'status': 'error', 'error': 'length_mismatch'}
    
    total_messages = sum(len(s) for s in sessions)
    start_time = time.time()
    
    print(f"[{question_idx+1}/{total}] 入库问题 {question_id}: {len(sessions)} 个会话, {total_messages} 条消息")
    
    try:
        # 按会话入库：每个会话 add + memorize
        # 这是 MFlow 官方推荐模式，确保 Episode Routing 能看到之前已入库的 Episode
        for session_idx, session in enumerate(sessions):
            date_str = dates[session_idx]
            session_text = format_session(session, date_str)
            
            await m_flow.add(
                data=session_text,
                dataset_name=dataset_name,
                created_at=parse_date(date_str)
            )
            
            # 每个会话单独 memorize，确保 Episode Routing 生效
            await m_flow.memorize(
                datasets=[dataset_name],
                content_type=ContentType.DIALOG,
                precise_mode=True,
            )
        
        elapsed = time.time() - start_time
        print(f"  ✓ 完成 ({elapsed:.1f}s)")
        
        return {
            'question_id': question_id,
            'status': 'success',
            'sessions': len(sessions),
            'messages': total_messages,
            'elapsed_seconds': elapsed
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ 失败: {e}")
        return {
            'question_id': question_id,
            'status': 'error',
            'error': str(e),
            'elapsed_seconds': elapsed
        }


async def main():
    parser = argparse.ArgumentParser(description='MFlow LongMemEval 入库脚本')
    parser.add_argument('--max-questions', type=int, default=50, help='最大处理问题数 (默认: 50)')
    parser.add_argument('--start-from', type=int, default=0, help='从第 N 个问题开始 (默认: 0)')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MFlow LongMemEval 入库脚本")
    print("=" * 60)
    print(f"MFlow 根目录: {MFLOW_ROOT}")
    print(f"数据路径: {DATA_PATH}")
    print(f"处理范围: 问题 {args.start_from} - {args.start_from + args.max_questions - 1}")
    print("=" * 60)
    
    # 加载数据
    questions = load_questions()
    
    # 选择处理范围
    end_idx = min(args.start_from + args.max_questions, len(questions))
    questions_to_process = questions[args.start_from:end_idx]
    
    print(f"\n开始入库 {len(questions_to_process)} 个问题...\n")
    
    results = []
    total_start = time.time()
    
    for idx, question in enumerate(questions_to_process):
        result = await ingest_question(question, args.start_from + idx, end_idx)
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    total_messages = sum(r.get('messages', 0) for r in results if r['status'] == 'success')
    
    print("\n" + "=" * 60)
    print("入库完成")
    print("=" * 60)
    print(f"成功: {success_count}, 失败: {error_count}")
    print(f"总消息数: {total_messages}")
    print(f"总耗时: {total_elapsed:.1f}s")
    if len(questions_to_process) > 0:
        print(f"平均: {total_elapsed/len(questions_to_process):.1f}s/问题")
    
    # 保存结果
    result_file = PROJECT_ROOT / "results" / f"mflow_ingest_results_{args.start_from}_{end_idx}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'start_idx': args.start_from,
            'end_idx': end_idx,
            'total_elapsed': total_elapsed,
            'success_count': success_count,
            'error_count': error_count,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {result_file}")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
