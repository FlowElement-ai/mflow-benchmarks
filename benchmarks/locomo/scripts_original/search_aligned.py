"""
M-flow LOCOMO Search + QA Script
Fully aligned with Mem0's evaluation/src/memzero/search.py

This script performs:
1. Memory retrieval for each question (both speakers)
2. Answer generation using aligned prompts
3. Result saving in Mem0-compatible format
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH

load_dotenv()


# Import M-flow
import m_flow
from m_flow.search.types import RecallMode


class MflowSearchAligned:
    """
    M-flow search implementation aligned with Mem0's MemorySearch class.
    
    Uses M-flow's standard search API (m_flow.search) instead of internal components.
    This ensures consistent behavior with M-flow's documented interface.
    """
    
    def __init__(
        self,
        output_path: str = "results.json",
        top_k: int = 10,
        dataset_prefix: str = "locomo_benchmark",
        use_graph: bool = False,
    ):
        """
        Initialize M-flow search.
        
        Args:
            output_path: Path to save results
            top_k: Number of memories to retrieve (default 10)
            dataset_prefix: Dataset name prefix
            use_graph: Whether to include graph relations (like Mem0's is_graph)
        """
        self.openai_client = OpenAI()
        self.top_k = top_k
        self.output_path = output_path
        self.results = defaultdict(list)
        self.dataset_prefix = dataset_prefix
        self.use_graph = use_graph
        
        # Select prompt based on graph mode
        self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH if use_graph else ANSWER_PROMPT
    
    async def search_memory(
        self,
        user_id: str,
        query: str,
        dataset_name: str,
    ) -> tuple[list[dict], list[dict], float]:
        """
        Search memories for a user using M-flow's EpisodicRetriever directly.
        
        NOTE: We bypass m_flow.search() API because it always returns string context,
        not Edge objects. We need Edge objects to extract Episode node attributes
        (summary, mentioned_time_start_ms) for Mem0-compatible output format.
        
        Returns:
            Tuple of (semantic_memories, graph_memories, search_time)
            - semantic_memories: List of {"memory": str, "timestamp": str, "score": float}
            - graph_memories: List of {"source": str, "relationship": str, "target": str}
            - search_time: Time taken for search
        """
        from m_flow.retrieval.episodic_retriever import EpisodicRetriever
        from m_flow.retrieval.episodic import EpisodicConfig
        from m_flow.context_global_variables import (
            backend_access_control_enabled,
            set_database_global_context_variables,
        )
        from m_flow.data.methods import get_datasets_by_name
        from m_flow.auth.methods import get_default_user
        
        start_time = time.time()
        semantic_memories = []
        graph_memories = []
        
        try:
            # Set dataset context for retrieval (only needed in multi-user mode)
            if backend_access_control_enabled():
                # Get the default user for benchmark context
                default_user = await get_default_user()
                
                # Look up dataset by name to get ID and owner_id
                datasets = await get_datasets_by_name(dataset_name, default_user.id)
                if datasets:
                    dataset = datasets[0]
                    await set_database_global_context_variables(dataset.id, dataset.owner_id)
                else:
                    print(f"Warning: Dataset '{dataset_name}' not found for user {default_user.id}")
            # In single-user mode, no context setup needed - uses default database
            
            # Use EpisodicRetriever directly to get raw triplets with Episode nodes
            config = EpisodicConfig(
                top_k=self.top_k,
                wide_search_top_k=self.top_k * 3,
                triplet_distance_penalty=3.5,
                display_mode="summary",  # Summary mode includes Episode nodes
            )
            retriever = EpisodicRetriever(
                top_k=self.top_k,
                config=config,
            )
            
            # get_triplets() returns raw Edge objects BEFORE display_mode filtering
            triplets = await retriever.get_triplets(query)
            
            if not triplets:
                return [], [], time.time() - start_time
            
            # Process triplets to extract Episode memories
            seen_episodes = set()
            
            for edge in triplets:
                # Check both nodes for Episode type
                for node in (edge.node1, edge.node2):
                    attrs = getattr(node, 'attributes', {}) or {}
                    if attrs.get("type") == "Episode":
                        node_id = str(getattr(node, 'id', ''))
                        if node_id in seen_episodes:
                            continue
                        seen_episodes.add(node_id)
                        
                        # Get timestamp (Mem0 format)
                        timestamp = self._extract_timestamp(attrs)
                        
                        # Get memory content (Episode summary)
                        summary = attrs.get("summary") or attrs.get("name") or ""
                        
                        if summary:
                            semantic_memories.append({
                                "memory": summary,
                                "timestamp": timestamp,
                                "score": round(getattr(edge, 'score', 0.9), 2),
                            })
                
                # Extract graph relations (if use_graph mode)
                if self.use_graph:
                    node1_attrs = getattr(edge.node1, 'attributes', {}) or {}
                    node2_attrs = getattr(edge.node2, 'attributes', {}) or {}
                    source = node1_attrs.get("name", str(getattr(edge.node1, 'id', '')))
                    target = node2_attrs.get("name", str(getattr(edge.node2, 'id', '')))
                    relationship = getattr(edge, 'edge_type', None) or "related_to"
                    
                    graph_memories.append({
                        "source": source,
                        "relationship": relationship,
                        "target": target,
                    })
            
            # Limit to top_k
            semantic_memories = semantic_memories[:self.top_k]
            
        except Exception as e:
            print(f"Search error for {user_id}: {e}")
            import traceback
            traceback.print_exc()
        
        end_time = time.time()
        return semantic_memories, graph_memories, end_time - start_time
    
    def _extract_timestamp(self, attrs: dict) -> str:
        """Extract timestamp from Episode's created_at (session date).

        Only uses created_at to avoid confusion from having two different
        time fields (mentioned_time vs created_at) visible to the LLM.
        """
        created_at = attrs.get("created_at")
        if created_at is not None:
            try:
                if isinstance(created_at, (int, float)):
                    ts = created_at / 1000 if created_at > 1e12 else created_at
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    return dt.strftime("%B %d, %Y")
                elif isinstance(created_at, datetime):
                    return created_at.strftime("%B %d, %Y")
                elif isinstance(created_at, str):
                    return created_at[:10]
            except (ValueError, OSError, TypeError):
                pass
        return "Unknown date"
    
    async def answer_question(
        self,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        dataset: str,
        question: str,
        answer: str,
        category: int,
    ) -> dict[str, Any]:
        """
        Answer a question using retrieved memories.
        
        M-flow architecture: Both speakers share the SAME dataset with objective facts.
        
        OPTIMIZATION: Single search instead of per-speaker search.
        Reasons:
        1. M-flow Episodes are objective fact summaries, not perspective-dependent
        2. Splitting search results into two groups doubles context length
        3. Increased context length reduces LLM accuracy
        4. The same facts are retrieved regardless of who is asking
        """
        # Single search - no need to search twice for the same dataset
        # This reduces context length and improves LLM accuracy
        memories, graph_relations, search_time = await self.search_memory(
            speaker_a_user_id, question, dataset  # user_id is only for context, not filtering
        )
        
        def format_memories(memories: list[dict]) -> str:
            if not memories:
                return "(No memories available)"
            lines = []
            for i, m in enumerate(memories, 1):
                ts = m.get('timestamp', 'Unknown')
                text = m.get('memory', '')
                lines.append(f"[Memory {i}] ({ts})\n{text}")
            return "\n\n".join(lines)
        
        def format_graph(relations: list[dict]) -> str:
            if not relations:
                return "[]"
            return json.dumps(relations, indent=4)
        
        # Generate answer using LLM with unified memories
        # Prompt preserves all original guidance from ANSWER_PROMPT but uses unified memories
        speaker_1_name = speaker_a_user_id.split("_")[0]
        speaker_2_name = speaker_b_user_id.split("_")[0]
        
        graph_section = ""
        if self.use_graph and graph_relations:
            graph_section = f"\nGraph Relations:\n\n{format_graph(graph_relations)}\n"

        prompt = f"""Answer the question using ONLY the memories below. Be concise (5-6 words). No explanation.

RULES:
- Read ALL memories before answering — do not stop at the first match.
- Timestamps = conversation session dates. Use them to resolve relative time references (e.g., "last year" in a May 2023 memory → 2022).
- If memories conflict, prefer the later one. One memory may be incomplete — combine evidence from multiple memories.
- Look for both direct mentions and indirect implications. Say "Unknown" only if nothing is relevant.

Memories from {speaker_1_name} & {speaker_2_name}:

{format_memories(memories)}
{graph_section}
Question: {question}

Answer:"""
        
        t1 = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,  # Deterministic output
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            print(f"  LLM error: {e}")
            response_text = f"[ERROR: {str(e)[:100]}]"
        t2 = time.time()
        
        return {
            "response": response_text,
            "memories": memories,  # Unified memories
            "num_memories": len(memories),
            "search_time": search_time,
            "graph_memories": graph_relations if self.use_graph else None,
            "response_time": t2 - t1,
            # Keep legacy fields for backward compatibility
            "speaker_1_memories": memories,
            "speaker_2_memories": [],
            "num_speaker_1_memories": len(memories),
            "num_speaker_2_memories": 0,
            "speaker_1_memory_time": search_time,
            "speaker_2_memory_time": 0,
        }
    
    async def process_question(
        self,
        val: dict,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        dataset: str,
    ) -> dict[str, Any]:
        """Process a single question (aligned with Mem0's process_question)."""
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")
        
        result = await self.answer_question(
            speaker_a_user_id,
            speaker_b_user_id,
            dataset,
            question,
            str(answer),  # Convert to string (some answers are integers)
            category,
        )
        
        result.update({
            "question": question,
            "answer": str(answer),
            "category": category,
            "evidence": evidence,
            "adversarial_answer": adversarial_answer,
        })
        
        return result
    
    async def process_data_file(self, file_path: str, max_conversations: int = None):
        """
        Process all questions from LOCOMO data file.
        Aligned with Mem0's process_data_file() method.
        
        Args:
            file_path: Path to LOCOMO JSON file
            max_conversations: Maximum conversations to process (for quick testing)
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Limit conversations if specified
        if max_conversations is not None:
            data = data[:max_conversations]
            print(f"Processing {len(data)} conversations (limited from original)...")
        else:
            print(f"Processing {len(data)} conversations...")
        
        for idx, item in enumerate(tqdm(data, desc="Conversations")):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            
            # User IDs match Mem0 format: "SpeakerName_idx"
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            
            # M-flow architecture: Both speakers share the same Episode-based dataset
            # This is correct because M-flow Episodes are objective fact summaries,
            # not perspective-dependent memories like Mem0.
            # Format: {prefix}_{speaker_a}_{speaker_b}_{idx}
            # Example: locomo_benchmark_Caroline_Melanie_0
            shared_dataset = f"{self.dataset_prefix}_{speaker_a}_{speaker_b}_{idx}"
            
            print(f"\n[Conv {idx}] {speaker_a} <-> {speaker_b}, {len(qa)} questions")
            print(f"  Dataset: {shared_dataset}")
            
            for question_item in tqdm(
                qa,
                total=len(qa),
                desc=f"  Questions",
                leave=False,
            ):
                try:
                    result = await self.process_question(
                        question_item,
                        speaker_a_user_id,
                        speaker_b_user_id,
                        shared_dataset,
                    )
                except Exception as e:
                    print(f"  Question error: {e}")
                    result = {
                        "question": question_item.get("question", ""),
                        "answer": str(question_item.get("answer", "")),
                        "category": question_item.get("category", -1),
                        "response": f"[ERROR: {str(e)[:100]}]",
                        "error": str(e),
                    }
                self.results[idx].append(result)
                
                # Save after each question (Mem0 behavior)
                with open(self.output_path, "w") as f:
                    json.dump(dict(self.results), f, indent=2)
        
        # Final save
        with open(self.output_path, "w") as f:
            json.dump(dict(self.results), f, indent=2)
        
        total_questions = sum(len(v) for v in self.results.values())
        print(f"\nComplete! {len(self.results)} conversations, {total_questions} questions")
        print(f"Results saved to: {self.output_path}")
        
        return dict(self.results)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run M-flow LOCOMO Search (aligned with Mem0)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../../dataset/locomo10.json",
        help="Path to LOCOMO JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./results/mflow_aligned_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of memories to retrieve (default 30)",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="locomo_benchmark",
        help="Dataset name prefix",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Include graph relations (like Mem0's is_graph mode)",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to process (for quick testing)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("M-flow LOCOMO Benchmark - Search Phase")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_path}")
    print(f"Top-K: {args.top_k}")
    print(f"Graph Mode: {args.use_graph}")
    if args.max_conversations:
        print(f"Max Conversations: {args.max_conversations}")
    print("=" * 60)
    
    searcher = MflowSearchAligned(
        output_path=args.output_path,
        top_k=args.top_k,
        dataset_prefix=args.dataset_prefix,
        use_graph=args.use_graph,
    )
    
    await searcher.process_data_file(args.data_path, max_conversations=args.max_conversations)


if __name__ == "__main__":
    asyncio.run(main())
