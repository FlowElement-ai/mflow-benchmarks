#!/usr/bin/env python3
"""
MFlow LongMemEval Benchmark - Results Analysis Script

Analyzes evaluation results and generates detailed reports.

Usage:
    python analyze_results.py [results_file]
    python analyze_results.py --compare mflow_results.json graphiti_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def load_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_single_results(data: Dict) -> Dict:
    """Analyze results from a single engine"""
    results = data.get('results', [])
    summary = data.get('summary', {})
    
    analysis = {
        'total_questions': len(results),
        'correct_count': sum(1 for r in results if r.get('llm_score', 0) == 1),
        'incorrect_count': sum(1 for r in results if r.get('llm_score', 0) == 0),
        'avg_bleu': summary.get('avg_bleu', 0),
        'avg_f1': summary.get('avg_f1', 0),
        'llm_accuracy': summary.get('llm_accuracy', 0),
        'avg_retrieval_ms': summary.get('avg_retrieval_ms', 0),
        'avg_generation_ms': summary.get('avg_generation_ms', 0),
    }
    
    # Score distribution
    bleu_scores = [r.get('bleu_score', 0) for r in results]
    f1_scores = [r.get('f1_score', 0) for r in results]
    
    analysis['bleu_distribution'] = {
        'min': min(bleu_scores) if bleu_scores else 0,
        'max': max(bleu_scores) if bleu_scores else 0,
        'perfect_matches': sum(1 for s in bleu_scores if s >= 0.99),
    }
    
    analysis['f1_distribution'] = {
        'min': min(f1_scores) if f1_scores else 0,
        'max': max(f1_scores) if f1_scores else 0,
        'high_overlap': sum(1 for s in f1_scores if s >= 0.5),
    }
    
    # Retrieval analysis
    retrieval_times = [r.get('retrieval_ms', 0) for r in results]
    analysis['retrieval_stats'] = {
        'min_ms': min(retrieval_times) if retrieval_times else 0,
        'max_ms': max(retrieval_times) if retrieval_times else 0,
        'median_ms': sorted(retrieval_times)[len(retrieval_times)//2] if retrieval_times else 0,
    }
    
    # Memory counts
    memory_counts = [r.get('memories_count', 0) for r in results]
    analysis['memory_stats'] = {
        'avg_memories': sum(memory_counts) / len(memory_counts) if memory_counts else 0,
        'zero_memories': sum(1 for c in memory_counts if c == 0),
    }
    
    # Error analysis
    incorrect = [r for r in results if r.get('llm_score', 0) == 0]
    analysis['error_examples'] = [
        {
            'question_id': r['question_id'],
            'question': r['question'][:80] + '...' if len(r['question']) > 80 else r['question'],
            'gold': r['gold_answer'],
            'generated': r['generated_answer'],
        }
        for r in incorrect[:5]  # First 5 errors
    ]
    
    # Perfect matches
    perfect = [r for r in results if r.get('bleu_score', 0) >= 0.99]
    analysis['perfect_examples'] = [
        {
            'question_id': r['question_id'],
            'question': r['question'][:80] + '...' if len(r['question']) > 80 else r['question'],
            'answer': r['gold_answer'],
        }
        for r in perfect[:5]
    ]
    
    return analysis


def compare_results(data1: Dict, data2: Dict, name1: str = "Engine1", name2: str = "Engine2") -> Dict:
    """Compare results from two engines"""
    results1 = data1.get('results', [])
    results2 = data2.get('results', [])
    
    # Build lookup by question_id
    r1_by_id = {r['question_id']: r for r in results1}
    r2_by_id = {r['question_id']: r for r in results2}
    
    common_ids = set(r1_by_id.keys()) & set(r2_by_id.keys())
    
    comparison = {
        'engine1': name1,
        'engine2': name2,
        'common_questions': len(common_ids),
        'only_in_1': len(set(r1_by_id.keys()) - common_ids),
        'only_in_2': len(set(r2_by_id.keys()) - common_ids),
    }
    
    # Per-question comparison
    both_correct = 0
    both_wrong = 0
    only_1_correct = 0
    only_2_correct = 0
    
    disagreements = []
    
    for qid in common_ids:
        r1 = r1_by_id[qid]
        r2 = r2_by_id[qid]
        
        score1 = r1.get('llm_score', 0)
        score2 = r2.get('llm_score', 0)
        
        if score1 == 1 and score2 == 1:
            both_correct += 1
        elif score1 == 0 and score2 == 0:
            both_wrong += 1
        elif score1 == 1:
            only_1_correct += 1
            disagreements.append({
                'question_id': qid,
                'question': r1['question'][:60] + '...',
                f'{name1}_answer': r1['generated_answer'],
                f'{name2}_answer': r2['generated_answer'],
                'gold': r1['gold_answer'],
                'winner': name1,
            })
        else:
            only_2_correct += 1
            disagreements.append({
                'question_id': qid,
                'question': r1['question'][:60] + '...',
                f'{name1}_answer': r1['generated_answer'],
                f'{name2}_answer': r2['generated_answer'],
                'gold': r1['gold_answer'],
                'winner': name2,
            })
    
    comparison['agreement'] = {
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        f'only_{name1}_correct': only_1_correct,
        f'only_{name2}_correct': only_2_correct,
        'agreement_rate': (both_correct + both_wrong) / len(common_ids) if common_ids else 0,
    }
    
    comparison['disagreements'] = disagreements[:10]  # First 10 disagreements
    
    # Metric comparison
    s1 = data1.get('summary', {})
    s2 = data2.get('summary', {})
    
    comparison['metrics'] = {
        'llm_accuracy': {
            name1: s1.get('llm_accuracy', 0),
            name2: s2.get('llm_accuracy', 0),
            'diff': s1.get('llm_accuracy', 0) - s2.get('llm_accuracy', 0),
        },
        'avg_bleu': {
            name1: s1.get('avg_bleu', 0),
            name2: s2.get('avg_bleu', 0),
            'diff': s1.get('avg_bleu', 0) - s2.get('avg_bleu', 0),
        },
        'avg_f1': {
            name1: s1.get('avg_f1', 0),
            name2: s2.get('avg_f1', 0),
            'diff': s1.get('avg_f1', 0) - s2.get('avg_f1', 0),
        },
        'avg_retrieval_ms': {
            name1: s1.get('avg_retrieval_ms', 0),
            name2: s2.get('avg_retrieval_ms', 0),
            'diff': s1.get('avg_retrieval_ms', 0) - s2.get('avg_retrieval_ms', 0),
        },
    }
    
    return comparison


def print_analysis(analysis: Dict, title: str = "Results Analysis") -> None:
    """Print analysis in readable format"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    
    print(f"\n## Overall Performance")
    print(f"  Total Questions: {analysis['total_questions']}")
    print(f"  Correct: {analysis['correct_count']} ({analysis['llm_accuracy']*100:.1f}%)")
    print(f"  Incorrect: {analysis['incorrect_count']}")
    
    print(f"\n## Metrics")
    print(f"  LLM-Judge Accuracy: {analysis['llm_accuracy']*100:.1f}%")
    print(f"  Average BLEU-1: {analysis['avg_bleu']:.4f}")
    print(f"  Average F1: {analysis['avg_f1']:.4f}")
    
    print(f"\n## Score Distribution")
    print(f"  BLEU-1 Range: {analysis['bleu_distribution']['min']:.4f} - {analysis['bleu_distribution']['max']:.4f}")
    print(f"  Perfect BLEU Matches: {analysis['bleu_distribution']['perfect_matches']}")
    print(f"  High F1 (>=0.5): {analysis['f1_distribution']['high_overlap']}")
    
    print(f"\n## Retrieval Performance")
    print(f"  Avg Retrieval Time: {analysis['avg_retrieval_ms']:.2f}ms")
    print(f"  Min/Max: {analysis['retrieval_stats']['min_ms']:.0f}ms / {analysis['retrieval_stats']['max_ms']:.0f}ms")
    print(f"  Median: {analysis['retrieval_stats']['median_ms']:.0f}ms")
    print(f"  Avg Memories Retrieved: {analysis['memory_stats']['avg_memories']:.1f}")
    print(f"  Zero-Memory Retrievals: {analysis['memory_stats']['zero_memories']}")
    
    if analysis.get('error_examples'):
        print(f"\n## Error Examples (first 5)")
        for i, ex in enumerate(analysis['error_examples'], 1):
            print(f"\n  {i}. [{ex['question_id']}]")
            print(f"     Q: {ex['question']}")
            print(f"     Gold: {ex['gold']}")
            print(f"     Generated: {ex['generated']}")
    
    if analysis.get('perfect_examples'):
        print(f"\n## Perfect Matches (first 5)")
        for i, ex in enumerate(analysis['perfect_examples'], 1):
            print(f"  {i}. [{ex['question_id']}] {ex['answer']}")


def print_comparison(comparison: Dict) -> None:
    """Print comparison in readable format"""
    print(f"\n{'='*70}")
    print(f"  Comparison: {comparison['engine1']} vs {comparison['engine2']}")
    print(f"{'='*70}")
    
    print(f"\n## Common Questions: {comparison['common_questions']}")
    
    print(f"\n## Agreement Analysis")
    a = comparison['agreement']
    print(f"  Both Correct: {a['both_correct']}")
    print(f"  Both Wrong: {a['both_wrong']}")
    e1_key = f"only_{comparison['engine1']}_correct"
    e2_key = f"only_{comparison['engine2']}_correct"
    print(f"  Only {comparison['engine1']} Correct: {a.get(e1_key, 0)}")
    print(f"  Only {comparison['engine2']} Correct: {a.get(e2_key, 0)}")
    print(f"  Agreement Rate: {a['agreement_rate']*100:.1f}%")
    
    print(f"\n## Metric Comparison")
    for metric, values in comparison['metrics'].items():
        e1 = values[comparison['engine1']]
        e2 = values[comparison['engine2']]
        diff = values['diff']
        better = comparison['engine1'] if diff > 0 else comparison['engine2']
        if metric == 'avg_retrieval_ms':
            better = comparison['engine1'] if diff < 0 else comparison['engine2']
        print(f"  {metric}:")
        print(f"    {comparison['engine1']}: {e1:.4f}")
        print(f"    {comparison['engine2']}: {e2:.4f}")
        print(f"    Better: {better} ({abs(diff):.4f} diff)")
    
    if comparison.get('disagreements'):
        print(f"\n## Disagreement Examples (first 10)")
        for i, d in enumerate(comparison['disagreements'], 1):
            print(f"\n  {i}. [{d['question_id']}] Winner: {d['winner']}")
            print(f"     Q: {d['question']}")
            print(f"     Gold: {d['gold']}")
            ans1_key = f"{comparison['engine1']}_answer"
            ans2_key = f"{comparison['engine2']}_answer"
            print(f"     {comparison['engine1']}: {d.get(ans1_key, 'N/A')}")
            print(f"     {comparison['engine2']}: {d.get(ans2_key, 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze MFlow benchmark results')
    parser.add_argument('results_file', nargs='?', help='Path to results JSON file')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                        help='Compare two result files')
    parser.add_argument('--output', '-o', help='Output file for analysis (JSON)')
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        data1 = load_results(args.compare[0])
        data2 = load_results(args.compare[1])
        
        name1 = data1.get('summary', {}).get('engine', 'Engine1')
        name2 = data2.get('summary', {}).get('engine', 'Engine2')
        
        comparison = compare_results(data1, data2, name1, name2)
        print_comparison(comparison)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to: {args.output}")
    
    elif args.results_file:
        # Single file analysis
        data = load_results(args.results_file)
        analysis = analyze_single_results(data)
        
        engine_name = data.get('summary', {}).get('engine', 'Unknown')
        print_analysis(analysis, f"{engine_name} Results Analysis")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to: {args.output}")
    
    else:
        # Try to find results files
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / 'results'
        
        mflow_file = results_dir / 'mflow_eval_results.json'
        graphiti_file = results_dir / 'graphiti_eval_results.json'
        
        if mflow_file.exists() and graphiti_file.exists():
            print("Found both MFlow and Graphiti results. Running comparison...")
            data1 = load_results(str(mflow_file))
            data2 = load_results(str(graphiti_file))
            comparison = compare_results(data1, data2, 'MFlow', 'Graphiti')
            print_comparison(comparison)
        elif mflow_file.exists():
            print(f"Found MFlow results at {mflow_file}")
            data = load_results(str(mflow_file))
            analysis = analyze_single_results(data)
            print_analysis(analysis, "MFlow Results Analysis")
        else:
            parser.print_help()
            print("\nNo results file specified and no default files found.")


if __name__ == '__main__':
    main()
