"""
Evaluation Metrics for M-flow LOCOMO Benchmark
Fully aligned with Mem0 official implementation
"""

import json
import re
from typing import Dict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI

from prompts import ACCURACY_PROMPT

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")


def calculate_bleu1(prediction: str, reference: str) -> float:
    """
    Calculate BLEU-1 score (unigram only).
    Fully aligned with Mem0's calculate_bleu_scores().
    
    Args:
        prediction: Generated answer
        reference: Gold answer
        
    Returns:
        BLEU-1 score between 0 and 1
    """
    if not prediction or not reference:
        return 0.0
    
    try:
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
    except Exception:
        # Fallback to simple split
        pred_tokens = prediction.lower().split()
        ref_tokens = [reference.lower().split()]
    
    if len(pred_tokens) == 0:
        return 0.0
    
    smooth = SmoothingFunction().method1
    
    try:
        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1, 0, 0, 0),  # BLEU-1: only unigram
            smoothing_function=smooth,
        )
    except Exception:
        score = 0.0
    
    return score


def calculate_f1(prediction: str, reference: str) -> float:
    """
    Calculate F1 score based on token overlap.
    Fully aligned with Mem0's calculate_metrics().
    
    Args:
        prediction: Generated answer
        reference: Gold answer
        
    Returns:
        F1 score between 0 and 1
    """
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    
    # Simple tokenization (aligned with Mem0's simple_tokenize)
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
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def extract_json(text: str) -> str:
    """Extract JSON object from text response."""
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        return match.group()
    return text


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    client: OpenAI = None,
    model: str = "gpt-4o-mini",
) -> int:
    """
    Use LLM to judge answer correctness.
    Fully aligned with Mem0's evaluate_llm_judge().
    
    Args:
        question: The question asked
        gold_answer: Ground truth answer
        generated_answer: Model's generated answer
        client: OpenAI client (creates one if None)
        model: Model to use for judgment
        
    Returns:
        1 if CORRECT, 0 if WRONG
    """
    if client is None:
        client = OpenAI()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0.0,  # Deterministic for consistency
        )
        
        result = json.loads(extract_json(response.choices[0].message.content))
        label = result.get("label", "WRONG").upper()
        return 1 if label == "CORRECT" else 0
        
    except Exception as e:
        print(f"LLM Judge error: {e}")
        return 0


def calculate_all_metrics(
    prediction: str,
    reference: str,
    question: str,
    client: OpenAI = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a single prediction.
    
    Args:
        prediction: Generated answer
        reference: Gold answer
        question: Original question
        client: OpenAI client
        model: Model for LLM-Judge
        
    Returns:
        Dictionary with bleu_score, f1_score, llm_score
    """
    return {
        "bleu_score": calculate_bleu1(prediction, reference),
        "f1_score": calculate_f1(prediction, reference),
        "llm_score": evaluate_llm_judge(question, reference, prediction, client, model),
    }
