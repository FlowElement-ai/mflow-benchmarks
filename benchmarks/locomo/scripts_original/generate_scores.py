"""
M-flow LOCOMO Score Generation Script
Fully aligned with Mem0's evaluation/generate_scores.py

This script generates the final benchmark report with:
- Per-category scores
- Overall scores
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    """Generate benchmark score report."""
    parser = argparse.ArgumentParser(
        description="Generate M-flow LOCOMO benchmark scores"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="./results/evaluation_metrics.json",
        help="Path to evaluation metrics file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./results/evaluation_report.csv",
        help="Path to save CSV report",
    )
    
    args = parser.parse_args()
    
    # Load evaluation metrics
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    # Flatten data into list
    all_items = []
    for key in data:
        all_items.extend(data[key])
    
    if not all_items:
        print("No evaluation data found.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_items)
    
    # Convert category to numeric
    df["category"] = pd.to_numeric(df["category"])
    
    print("=" * 70)
    print("M-flow LOCOMO Benchmark Results")
    print("(Aligned with Mem0 Evaluation Methodology)")
    print("=" * 70)
    
    # Calculate mean scores by category
    category_stats = df.groupby("category").agg({
        "bleu_score": "mean",
        "f1_score": "mean",
        "llm_score": "mean",
    }).round(4)
    
    # Add count
    category_stats["count"] = df.groupby("category").size()
    
    # Rename columns for display
    category_stats.columns = ["BLEU-1", "F1", "LLM-J", "Count"]
    
    print("\nPer-Category Scores:")
    print("-" * 70)
    print(category_stats.to_string())
    
    # Calculate overall means
    overall = {
        "BLEU-1": round(df["bleu_score"].mean(), 4),
        "F1": round(df["f1_score"].mean(), 4),
        "LLM-J (%)": round(df["llm_score"].mean() * 100, 2),
        "Total Questions": len(df),
    }
    
    print("\n" + "-" * 70)
    print("Overall Scores:")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    
    print("=" * 70)
    
    # Save CSV report
    category_stats.to_csv(args.output_csv)
    print(f"\nReport saved to: {args.output_csv}")
    
    # Also save a detailed JSON summary
    summary = {
        "per_category": category_stats.to_dict(),
        "overall": overall,
    }
    
    summary_path = Path(args.output_csv).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Print comparison table format (for easy copy-paste)
    print("\n" + "=" * 70)
    print("Comparison Format (copy for comparison with Mem0):")
    print("-" * 70)
    print("| Technique | BLEU-1 | F1 | LLM-J (%) |")
    print("|-----------|--------|-----|-----------|")
    print(f"| M-flow (Episodic) | {overall['BLEU-1']:.4f} | {overall['F1']:.4f} | {overall['LLM-J (%)']:.1f} |")
    print("=" * 70)


if __name__ == "__main__":
    main()
