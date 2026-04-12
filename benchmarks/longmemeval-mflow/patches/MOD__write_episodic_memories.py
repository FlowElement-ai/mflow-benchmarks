# m_flow/memory/episodic/write_episodic_memories.py
"""
Episodic Memory write task (cross-batch incremental update)

Converts FragmentDigest batch to Episode + Facet + rich semantic edges,
then passes to persist_memory_nodes for persistence to graph database and vector index.

Features:
- Cross-batch incremental update: Ingestion retrieval routing (Episode Router)
- Group writes by target_episode_id to avoid conflicts with multiple Episode nodes in same batch
- Semantic merge, aliases, review, entity selection
- Facet.aliases_text participates in indexing (fallback recall)
- LLM outputs aliases + alias_updates
- Alias filtering + optional semantic synonym merging
- Unified write logic: alias_updates application + string/semantic merging + aliases_text writing

Design principles (aligned with brute force triplet search success pattern):
- Episode.summary analogous to ContentFragment.text (high semantic density anchor)
- Facet.search_text analogous to Entity.name (short sharp anchor)
- Facet.aliases_text as fallback recall entry
- has_facet / involves_entity edge_text analogous to contains.edge_text (rich semantic links)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from m_flow.adapters.graph.graph_db_interface import GraphProvider
    from m_flow.adapters.vector.vector_db_interface import VectorProvider
    from m_flow.core.domain.models import Procedure

from m_flow.context_global_variables import current_dataset_id
from m_flow.shared.logging_utils import get_logger
from m_flow.adapters.graph import get_graph_provider
from m_flow.adapters.vector import get_vector_provider

from m_flow.core.domain.models import (
    Entity,
    Episode,
    Facet,
)  # Entity is new, Entity is alias
from m_flow.core.domain.models.memory_space import MemorySpace
from m_flow.core.domain.utils.generate_node_id import generate_node_id

from m_flow.knowledge.summarization.models import FragmentDigest
from m_flow.memory.episodic.models import (
    RoutingType,
)
from m_flow.memory.episodic.sentence_level_routing import (
    has_v2_routing,
    get_sentence_classifications,
)
from m_flow.memory.procedural.write_procedural_memories import _compile_and_build_procedure
from m_flow.memory.episodic.episodic_ingestion_config import (
    merge_config_with_params,
)

# ============================================================
# Import from modularized files (Step 3D: large-scale modularization)
# ============================================================

# Context variable management

# Text normalization functions
from m_flow.memory.episodic.normalization import (
    truncate as _truncate,
)

# Edge text generation functions

from m_flow.memory.episodic.state import (
    fetch_episode_state,
)

from m_flow.memory.episodic.llm_tasks import (
    _get_tracker,
)


# Pure functions extracted to utils/pure_functions.py (Phase 1 refactoring)
# Re-export all for backward compatibility with existing tests
from m_flow.memory.episodic.utils.pure_functions import (  # noqa: F401
    _episode_sort_key,
    _extract_time_fields_from_episode,
    _extract_chunk_summaries_from_text_summaries,
    _split_long_summary,
    _create_facets_from_sections,
    _extract_all_sections_from_summaries,
    _has_valid_sections,
    _extract_event_sentences,
    _create_facets_from_sections_direct,
    _generate_episode_summary_from_sections,
    _extract_entities_from_chunk,
    ensure_nodeset,
    _choose_better_description,
    _create_same_entity_as_edges,
)

# Data classes extracted to utils/models.py (Phase 2 refactoring)

# Entity lookup functions extracted to utils/entity_lookup.py (Phase 2 refactoring)
# Re-export for backward compatibility
from m_flow.memory.episodic.utils.entity_lookup import (  # noqa: F401
    find_existing_entities_by_canonical_name as _find_existing_entities_by_canonical_name,
    batch_find_existing_entities_by_canonical_names as _batch_find_existing_entities_by_canonical_names,
)
from m_flow.memory.episodic.utils.models import RoutingResult  # noqa: F401

from m_flow.memory.episodic.routing.document_router import (
    route_documents_to_episodes as _route_documents_to_episodes,
)

# Phase 4-New: Episode builder pipeline modules
from m_flow.memory.episodic.episode_builder import (
    EpisodeConfig,
    EpisodeContext,
    execute_phase0a,
    execute_phase0c,
    execute_step1,
    execute_step2,
    _build_has_facet_edges,
    _build_involves_entity_edges,
    _queue_facet_entity_edges,
    _collect_same_entity_as_edges,
    _build_includes_chunk_edges,
    _create_episode,
)

logger = get_logger("episodic.write")

# ============================================================
# Global LLM concurrency control (shared across ALL concurrent
# write_episodic_memories calls within the same process)
# ============================================================
import asyncio as _asyncio_global

# Note: _MIN_CONCURRENCY_FOR_PARALLEL is now imported from m_flow.memory.episodic.routing.document_router

# -----------------------------
# Helpers: env (imported from env_utils.py)
# -----------------------------

# Note: _truncate, _nfkc, normalize_for_compare, normalize_for_id 
# are now imported from m_flow.memory.episodic.normalization

# Note: _episode_sort_key, _extract_time_fields_from_episode, _SECTION_PATTERNS,
# _split_long_summary, _extract_chunk_summaries_from_text_summaries, _create_facets_from_sections,
# _extract_all_sections_from_summaries, _has_valid_sections, _extract_event_sentences,
# _create_facets_from_sections_direct, _generate_episode_summary_from_sections,
# _extract_entities_from_chunk, ensure_nodeset, _choose_better_description, _create_same_entity_as_edges
# are now imported from m_flow.memory.episodic.utils.pure_functions

# Note: RoutingResult, _FacetUpdate are now imported from m_flow.memory.episodic.utils.models

# Note: _find_existing_entities_by_canonical_name, _batch_find_existing_entities_by_canonical_names
# are now imported from m_flow.memory.episodic.utils.entity_lookup

# Note: _create_same_entity_as_edges is now imported from m_flow.memory.episodic.utils.pure_functions

# Note: Edge text generators (_make_has_facet_edge_text, _make_involves_entity_edge_text, etc.)
# are now imported from m_flow.memory.episodic.edge_text_generators


# ============================================================
# Routing decision functions
# ============================================================
# Note: _route_documents_to_episodes is now imported from
# m_flow.memory.episodic.routing.document_router


# -----------------------------
# Main: write_episodic_memories
# -----------------------------


async def write_episodic_memories(
    summaries: List[FragmentDigest],
    *,
    episodic_nodeset_name: str = "Episodic",
    max_entities_per_episode: int = 0,  # 0 = unlimited
    max_new_facets_per_batch: int = 20,
    max_existing_facets_in_prompt: int = 60,
    max_chunk_summaries_in_prompt: int = 40,
    max_candidate_entities_in_prompt: int = 25,
    max_aliases_per_facet: int = 10,
    aliases_text_max_chars: int = 400,
    evidence_chunks_per_facet: int = 3,
    enable_semantic_merge: Optional[bool] = None,
    semantic_merge_threshold: Optional[float] = None,
    enable_episode_routing: Optional[bool] = None,
    # FacetPoint parameters
    enable_facet_points: Optional[bool] = None,
    facet_points_prompt_file: Optional[str] = None,
    max_point_aliases_text_chars: int = 400,
    # LLM entity extraction for routing
    enable_llm_entity_for_routing: Optional[bool] = None,
    # Unified episodic + procedural routing
    enable_procedural_routing: bool = False,
    procedural_decisions_out: Optional[List[dict]] = None,
    precise_mode: bool = False,
) -> List[Any]:
    """
    Episodic write (cross-batch incremental update):

    Features:
    - Ingestion retrieval routing: new docs can be routed to existing Episodes (cross-batch incremental update)
    - Group by target_episode_id: avoid conflicts with multiple Episode nodes in same batch
    - Incremental update Episode summary/title/signature (stable)
    - New facets + alias_updates
    - String deduplication merging + (optional) semantic synonym merging
    - "Synonym expressions" from merging are not discarded, absorbed into aliases
    - Write aliases_text to participate in indexing (Facet_search_text + Facet_aliases_text)
    - Write evidence edges: Facet--supported_by-->ContentFragment, Episode--includes_chunk-->ContentFragment
    - Entity memory_spaces (ensures episodic subgraph projection can include entity edges)

    Args:
        summaries: FragmentDigest list (from compress_text)
        episodic_nodeset_name: MemorySpace name
        max_entities_per_episode: Maximum number of Entities per Episode (0 = unlimited)
        max_new_facets_per_batch: Maximum number of new Facets per batch
        max_existing_facets_in_prompt: Maximum number of existing facets in prompt
        max_chunk_summaries_in_prompt: Maximum number of chunk summaries in prompt
        max_candidate_entities_in_prompt: Maximum number of candidate entities in prompt
        max_aliases_per_facet: Maximum number of aliases per Facet
        aliases_text_max_chars: Maximum characters for aliases_text
        evidence_chunks_per_facet: Number of evidence chunks per Facet
        enable_semantic_merge: Whether to enable semantic synonym merging
        semantic_merge_threshold: Semantic merge threshold
        enable_episode_routing: Whether to enable ingestion retrieval routing (cross-batch incremental update)

    Returns:
        List of supplemented MemoryNodes
    """
    if not summaries:
        return summaries

    # Merge function parameters with environment variable configuration
    cfg = merge_config_with_params(
        None,  # Read base configuration from environment variables
        enable_semantic_merge=enable_semantic_merge,
        semantic_merge_threshold=semantic_merge_threshold,
        enable_episode_routing=enable_episode_routing,
        enable_facet_points=enable_facet_points,
        enable_llm_entity_for_routing=enable_llm_entity_for_routing,
        episodic_nodeset_name=episodic_nodeset_name,
        max_entities_per_episode=max_entities_per_episode,
        max_new_facets_per_batch=max_new_facets_per_batch,
        max_existing_facets_in_prompt=max_existing_facets_in_prompt,
        max_chunk_summaries_in_prompt=max_chunk_summaries_in_prompt,
        max_candidate_entities_in_prompt=max_candidate_entities_in_prompt,
        max_aliases_per_facet=max_aliases_per_facet,
        aliases_text_max_chars=aliases_text_max_chars,
        evidence_chunks_per_facet=evidence_chunks_per_facet,
        facet_points_prompt_file=facet_points_prompt_file,
        max_point_aliases_text_chars=max_point_aliases_text_chars,
    )
    
    # Use config values
    enable_semantic_merge = cfg.enable_semantic_merge
    semantic_merge_threshold = cfg.semantic_merge_threshold
    enable_episode_routing = cfg.enable_episode_routing
    enable_facet_points = cfg.enable_facet_points
    enable_llm_entity_for_routing = cfg.enable_llm_entity_for_routing
    facet_points_prompt_file = cfg.facet_points_prompt_file
    episodic_nodeset_name = cfg.episodic_nodeset_name
    max_entities_per_episode = cfg.max_entities_per_episode
    max_new_facets_per_batch = cfg.max_new_facets_per_batch
    max_existing_facets_in_prompt = cfg.max_existing_facets_in_prompt
    max_chunk_summaries_in_prompt = cfg.max_chunk_summaries_in_prompt
    max_candidate_entities_in_prompt = cfg.max_candidate_entities_in_prompt
    max_aliases_per_facet = cfg.max_aliases_per_facet
    aliases_text_max_chars = cfg.aliases_text_max_chars
    evidence_chunks_per_facet = cfg.evidence_chunks_per_facet
    max_point_aliases_text_chars = cfg.max_point_aliases_text_chars

    # ============================================================
    # Content Routing: Process ALL content (both Episodic and Atomic)
    # Supports both V1 (FragmentDigest.routing_type) and V2 (chunk.metadata) routing
    # Atomic content now also creates Episodes for full graph structure
    # ============================================================
    
    # Check for V1 routing (FragmentDigest.routing_type)
    has_v1_routing = any(getattr(s, "routing_type", None) is not None for s in summaries)
    
    # Check for sentence-level routing (chunk.metadata["sentence_classifications"])
    has_sentence_routing = any(has_v2_routing(s.made_from) for s in summaries)
    
    if has_sentence_routing:
        # V2 sentence-level routing
        # Process ALL chunks with sentence classifications (both episodic AND atomic)
        # Atomic sentences now also create Episodes for full graph structure
        
        summaries_to_process = [
            s
            for s in summaries
            if get_sentence_classifications(s.made_from)  # Has any classified sentences
        ]
        
        if not summaries_to_process:
            logger.info("[episodic] No classified content, skipping Episode creation")
            return list(summaries)
        
        # Count episodic vs atomic for logging
        episodic_count = sum(
            1
            for s in summaries
            for c in get_sentence_classifications(s.made_from)
            if c.get("routing_type") == "episodic"
        )
        atomic_count = sum(
            1
            for s in summaries
            for c in get_sentence_classifications(s.made_from)
            if c.get("routing_type") == "atomic"
        )
        
        logger.info(
            f"[episodic] Content routing V2 enabled: processing "
            f"{len(summaries_to_process)}/{len(summaries)} chunks "
            f"({episodic_count} episodic + {atomic_count} atomic sentences → Episodes)"
        )
    elif has_v1_routing:
        # V1 chunk-level routing
        # Now processes ALL content (both Episodic and Atomic) as Episodes
        episodic_count = sum(
            1 for s in summaries if getattr(s, "routing_type", None) != RoutingType.ATOMIC
        )
        atomic_count = sum(
            1 for s in summaries if getattr(s, "routing_type", None) == RoutingType.ATOMIC
        )
        
        summaries_to_process = summaries  # Process ALL content
        
        logger.info(
            f"[episodic] Content routing V1 enabled: processing ALL "
            f"{len(summaries_to_process)} summaries ({episodic_count} episodic + {atomic_count} atomic → Episodes)"
        )
    else:
        # No routing, process all content
        summaries_to_process = summaries

    # deterministic MemorySpace id
    episodic_nodeset = MemorySpace(
        id=generate_node_id(f"MemorySpace:{episodic_nodeset_name}"),
        name=episodic_nodeset_name,
    )

    # ============================================================
    # Group summaries for Episode creation
    # V2: Group by event_id (from sentence_classifications)
    # V1 mode: Group by document_id
    # ============================================================
    by_doc: Dict[str, List[FragmentDigest]] = {}
    
    if has_sentence_routing:
        # Sentence-level mode: Group by event_id to allow multiple Episodes from same chunk
        # Processes BOTH episodic AND atomic sentences (atomic also creates Episodes)
        
        event_to_summaries: Dict[str, List[FragmentDigest]] = {}
        event_topics: Dict[str, str] = {}  # event_id -> suggested_topic
        event_routing_types: Dict[str, str] = {}  # event_id -> "episodic" or "atomic"
        
        for s in summaries_to_process:
            chunk = s.made_from
            classifications = get_sentence_classifications(chunk)
            
            # Group by event_id - now includes BOTH episodic and atomic
            seen_events = set()
            for c in classifications:
                event_id = c.get("event_id")
                routing_type = c.get("routing_type", "episodic")
                
                # All sentences with event_id (both episodic and atomic) are processed
                if event_id and event_id not in seen_events:
                    seen_events.add(event_id)
                    event_to_summaries.setdefault(event_id, []).append(s)
                    # Store topic for Episode name generation
                    if event_id not in event_topics:
                        event_topics[event_id] = (
                            c.get("event_topic") or c.get("suggested_topic") or ""
                        )
                    # Track routing type for potential differentiation
                    if event_id not in event_routing_types:
                        event_routing_types[event_id] = routing_type
            
            # Fallback: if no events found, use document grouping
            if not seen_events:
                doc = getattr(chunk, "is_part_of", None)
                doc_id = str(getattr(doc, "id", "")) or "__unknown_doc__"
                event_to_summaries.setdefault(f"doc_{doc_id}", []).append(s)
        
        by_doc = event_to_summaries
        
        # Log breakdown
        episodic_events = sum(1 for rt in event_routing_types.values() if rt == "episodic")
        atomic_events = sum(1 for rt in event_routing_types.values() if rt == "atomic")
        logger.info(
            f"[episodic] V2 grouping: {len(by_doc)} events "
            f"({episodic_events} episodic + {atomic_events} atomic) "
            f"from {len(summaries_to_process)} chunks"
        )
    else:
        # V1 mode: Group by document_id
        for s in summaries_to_process:
            chunk = s.made_from
            doc = getattr(chunk, "is_part_of", None)
            doc_id = str(getattr(doc, "id", "")) or "__unknown_doc__"
            by_doc.setdefault(doc_id, []).append(s)

    graph_engine = await get_graph_provider()
    vector_engine = get_vector_provider()

    out: List[Any] = []
    out.append(episodic_nodeset)
    
    # P2 Optimization: Collect procedural compile tasks for parallel execution
    # Tasks are started after concept extraction and run in parallel with Step 2
    _procedural_compile_tasks: List[_asyncio_global.Stage] = []

    # Step 2: Ingestion flow logging - batch-level logger
    batch_id = f"B{int(time.time() * 1000) % 100000}" if "time" in dir() else f"B{len(summaries)}"
    logger.info(
        f"[{batch_id}] 📥 Episodic ingestion started: {len(summaries)} summaries, {len(by_doc)} docs"
    )

    # ============================================================
    # Ingestion-time routing
    # ============================================================
    
    # Pass event_routing_types if available (sentence-level mode)
    # For document mode, this will be empty and routing_type will be determined by other means
    _event_routing_types_for_router = event_routing_types if has_sentence_routing else {}
    
    # Get dataset_id from ContextVar for Episode Routing isolation
    _dataset_id = current_dataset_id.get()
    
    routing_result = await _route_documents_to_episodes(
        by_doc=by_doc,
        graph_engine=graph_engine,
        vector_engine=vector_engine,
        enable_episode_routing=enable_episode_routing,
        enable_llm_entity_for_routing=enable_llm_entity_for_routing,
        max_entities_per_episode=max_entities_per_episode,
        max_chunk_summaries_in_prompt=max_chunk_summaries_in_prompt,
        event_routing_types=_event_routing_types_for_router,
        target_nodeset_id=str(episodic_nodeset.id),  # Dataset isolation (graph level)
        target_dataset_id=_dataset_id,  # Dataset isolation (vector filter)
    )
    
    # Destructure routing result
    by_episode = routing_result.by_episode
    episode_doc_titles = routing_result.episode_doc_titles
    doc_entity_cache = routing_result.doc_entity_cache
    original_event_routing_types = routing_result.original_event_routing_types
    routing_decisions = routing_result.routing_decisions
    episode_memory_types = routing_result.episode_memory_types
    episode_source_events = routing_result.episode_source_events  # Event-Level Sections
    
    # Counter for same_entity_as edges (for logging)
    # The actual edges are stored in _pending_same_entity_edges via context_vars
    same_entity_as_edge_count: int = 0

    # ============================================================
    # Now write per target episode_id (one Episode node per id per batch)
    # ============================================================

    for episode_id_str in sorted(by_episode.keys()):
        doc_summaries_raw = sorted(by_episode[episode_id_str], key=_episode_sort_key)

        # BUG FIX: Deduplication - ensure each chunk appears only once
        # Problem: When V2 Content Routing routes multiple events from the same chunk to one episode,
        #          by_episode[episode_id] contains multiple FragmentDigest references pointing to the same chunk
        # Impact: _extract_event_sentences would extract duplicate sentences, causing LLM to output repeated "Block 1/2/3/4" content
        seen_chunk_ids: set = set()
        doc_summaries: List[FragmentDigest] = []
        for ts in doc_summaries_raw:
            if ts.made_from is None:
                doc_summaries.append(ts)
                continue
            chunk_id = id(ts.made_from)
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                doc_summaries.append(ts)

        if len(doc_summaries) < len(doc_summaries_raw):
            logger.debug(
                f"[episodic] Deduped doc_summaries: {len(doc_summaries_raw)} → {len(doc_summaries)} "
                f"(removed {len(doc_summaries_raw) - len(doc_summaries)} duplicate chunk refs)"
            )
        
        # P2 Optimization: Collect procedural candidates for this episode
        # These will be used to start compile tasks after concept extraction
        _current_episode_candidates: List[dict] = []

        # Merge doc titles if multiple docs routed to same episode
        titles = episode_doc_titles.get(episode_id_str, [])
        if titles:
            doc_title = titles[0]
            if len(titles) > 1:
                doc_title = f"{_truncate(doc_title, 50)} (+{len(titles) - 1} docs)"
        else:
            doc_title = "Document"

        # fetch existing episode state
        state = await fetch_episode_state(graph_engine, episode_id_str)


        # existing facets lines for prompt
        existing_facets_lines = []
        for f in state.facets[:max_existing_facets_in_prompt]:
            st = _truncate(f.search_text, 80)
            ft = _truncate(f.facet_type, 40)
            if st:
                existing_facets_lines.append(f"- {ft} | {st}")

        # chunk summaries for prompt (using sections if available, no truncation)
        # When sections are available, each section becomes a separate chunk summary item,
        # ensuring complete coverage of all subtopics for facet generation.
        chunk_summaries = _extract_chunk_summaries_from_text_summaries(
            doc_summaries,
            max_items=max_chunk_summaries_in_prompt,
        )

        # ============================================================
        # Process episode using pipeline
        # Phase 4-New: All episode processing is now done via pipeline
        # ============================================================

        # Build EpisodeConfig
        config = _build_episode_config(
            max_entities_per_episode=max_entities_per_episode,
            max_candidate_entities_in_prompt=max_candidate_entities_in_prompt,
            max_new_facets_per_batch=max_new_facets_per_batch,
            max_existing_facets_in_prompt=max_existing_facets_in_prompt,
            max_aliases_per_facet=max_aliases_per_facet,
            aliases_text_max_chars=aliases_text_max_chars,
            evidence_chunks_per_facet=evidence_chunks_per_facet,
            max_chunk_summaries_in_prompt=max_chunk_summaries_in_prompt,
            enable_semantic_facet_merge=enable_semantic_merge,
            semantic_merge_threshold=semantic_merge_threshold,
            enable_episode_routing=enable_episode_routing,
            enable_facet_points=enable_facet_points,
            enable_llm_entity_for_routing=enable_llm_entity_for_routing,
            enable_procedural_routing=enable_procedural_routing,
            facet_points_prompt_file=facet_points_prompt_file,
            max_point_aliases_text_chars=max_point_aliases_text_chars,
            content_routing_disabled=not has_sentence_routing,
            precise_mode=precise_mode,
        )

        # Build EpisodeContext
        ctx = await _build_episode_context(
            episode_id_str=episode_id_str,
            doc_summaries=doc_summaries,
            chunk_summaries=chunk_summaries,
            state=state,
            doc_title=doc_title,
            doc_entity_cache=doc_entity_cache,
            episode_source_events=episode_source_events,
            original_event_routing_types=original_event_routing_types,
            routing_decisions=routing_decisions,
            episode_memory_types=episode_memory_types,
            config=config,
            graph_engine=graph_engine,
            vector_engine=vector_engine,
            episodic_nodeset=episodic_nodeset,
            batch_id=batch_id,
            dataset_id=_dataset_id,
        )

        # Process episode using pipeline
        (
            episode,
            procedural_tasks,
            same_entity_count,
            concept_types,
        ) = await _process_single_episode_pipeline(
            ctx=ctx,
            existing_facets_lines=existing_facets_lines,
        )

        # Collect results
        if episode:
            out.append(episode)
        # Add EntityType objects to output (for graph storage)
        for ct in concept_types:
            out.append(ct)
        _procedural_compile_tasks.extend(procedural_tasks)
        same_entity_as_edge_count += same_entity_count

    logger.info(
        f"[episodic] Generated {len([x for x in out if isinstance(x, Episode)])} episodes "
        f"from {len(summaries)} summaries"
    )
    
    # [TIME_PROPAGATION_LOG] Batch summary statistics
    # Count Episodes
    episodes_with_time = sum(
        1 for x in out if isinstance(x, Episode) and x.mentioned_time_start_ms is not None
    )
    total_episodes = sum(1 for x in out if isinstance(x, Episode))
    
    # Count Facets and Entities - need to extract from Episode's edge relationships
    all_facets = []
    all_entities = []
    for x in out:
        if isinstance(x, Episode):
            # Extract Facets from has_facet edges
            if x.has_facet:
                for _edge, facet in x.has_facet:  # _edge unused but required for unpacking
                    if isinstance(facet, Facet):
                        all_facets.append(facet)
            # Extract Entities from involves_entity edges
            if x.involves_entity:
                for _edge, entity in x.involves_entity:  # _edge unused but required for unpacking
                    if isinstance(entity, Entity):
                        all_entities.append(entity)
    
    facets_with_time = sum(
        1 for f in all_facets if getattr(f, "mentioned_time_start_ms", None) is not None
    )
    entities_with_time = sum(
        1 for e in all_entities if getattr(e, "mentioned_time_start_ms", None) is not None
    )
    
    logger.info(
        f"[TIME_PROPAGATION] [STATS] Batch summary: "
        f"Episodes with time: {episodes_with_time}/{total_episodes}, "
        f"Facets with time: {facets_with_time}/{len(all_facets)}, "
        f"Entities with time: {entities_with_time}/{len(all_entities)}"
    )

    # Output LLM call statistics
    tracker = _get_tracker()
    tracker.log_summary()

    # ============================================================
    # same_entity_as edges: queued for later insertion (after persist_memory_nodes)
    # The edges are stored in _pending_same_entity_edges global queue
    # and will be written by write_same_entity_edges task
    # ============================================================
    if same_entity_as_edge_count > 0:
        logger.info(
            f"[episodic] Queued {same_entity_as_edge_count} same_entity_as edges "
            f"for post-processing (will be written after persist_memory_nodes)"
        )

    # ============================================================
    # P2 Optimization: Wait for all procedural compile tasks to complete
    # ============================================================
    if _procedural_compile_tasks:
        _proc_bridge_wait_start = time.time()
        logger.info(
            f"[episodic.procedural_bridge] Waiting for {len(_procedural_compile_tasks)} procedural compile tasks..."
        )

        procedural_results = await _asyncio_global.gather(
            *_procedural_compile_tasks, return_exceptions=True
        )
        
        # Collect successful results
        procedural_success_count = 0
        procedural_fail_count = 0
        for result in procedural_results:
            if isinstance(result, Exception):
                logger.warning(f"[episodic.procedural_bridge] Procedural compile failed: {result}")
                procedural_fail_count += 1
            elif result is not None:
                out.append(result)
                procedural_success_count += 1
        
        _proc_bridge_wait_elapsed = time.time() - _proc_bridge_wait_start
        logger.info(
            f"[episodic.procedural_bridge] Procedural compile complete: {_proc_bridge_wait_elapsed:.2f}s, "
            f"success={procedural_success_count}, failed={procedural_fail_count}"
        )
    
    # Step 2: Ingestion process log - batch completion summary
    logger.info(
        f"[{batch_id}] [OUT] Episodic ingestion complete: "
        f"episodes={len(by_episode)}, datapoints={len(out)}"
        )
    
    return out


# Note: write_same_entity_edges and write_facet_entity_edges
# are now imported from m_flow.memory.episodic.edge_writers


# ============================================================
# Phase 4-New-G: Pipeline Coordination Functions
# These functions use the modularized episode_builder modules
# to provide a cleaner, more maintainable implementation.
# ============================================================


def _build_episode_config(
    max_entities_per_episode: int,
    max_candidate_entities_in_prompt: int,
    max_new_facets_per_batch: int,
    max_existing_facets_in_prompt: int,
    max_aliases_per_facet: int,
    aliases_text_max_chars: int,
    evidence_chunks_per_facet: int,
    max_chunk_summaries_in_prompt: int,
    enable_semantic_facet_merge: bool,
    semantic_merge_threshold: float,
    enable_episode_routing: bool,
    enable_facet_points: bool,
    enable_llm_entity_for_routing: bool,
    enable_procedural_routing: bool,
    facet_points_prompt_file: Optional[str],
    max_point_aliases_text_chars: int = 400,
    content_routing_disabled: bool = False,
    precise_mode: bool = False,
) -> EpisodeConfig:
    """
    Build EpisodeConfig from individual parameters.

    This helper function converts the many configuration parameters
    from write_episodic_memories into a single EpisodeConfig object.

    Phase 4-New-G: Helper for pipeline integration.
    """
    return EpisodeConfig(
        max_entities_per_episode=max_entities_per_episode,
        max_candidate_entities_in_prompt=max_candidate_entities_in_prompt,
        max_new_facets_per_batch=max_new_facets_per_batch,
        max_existing_facets_in_prompt=max_existing_facets_in_prompt,
        max_aliases_per_facet=max_aliases_per_facet,
        aliases_text_max_chars=aliases_text_max_chars,
        evidence_chunks_per_facet=evidence_chunks_per_facet,
        max_chunk_summaries_in_prompt=max_chunk_summaries_in_prompt,
        enable_semantic_merge=enable_semantic_facet_merge,
        semantic_merge_threshold=semantic_merge_threshold,
        enable_episode_routing=enable_episode_routing,
        enable_facet_points=enable_facet_points,
        enable_llm_entity_for_routing=enable_llm_entity_for_routing,
        enable_procedural_routing=enable_procedural_routing,
        facet_points_prompt_file=facet_points_prompt_file,
        max_point_aliases_text_chars=max_point_aliases_text_chars,
        content_routing_disabled=content_routing_disabled,
        precise_mode=precise_mode,
    )


async def _build_episode_context(
    episode_id_str: str,
    doc_summaries: List[FragmentDigest],
    chunk_summaries: List[str],
    state: Any,  # EpisodeState
    doc_title: str,
    doc_entity_cache: Dict[str, List[str]],
    episode_source_events: Dict[str, List[str]],
    original_event_routing_types: Dict[str, str],
    routing_decisions: Dict[str, str],
    episode_memory_types: Dict[str, str],
    config: EpisodeConfig,
    graph_engine: "GraphProvider",
    vector_engine: "VectorProvider",
    episodic_nodeset: Any,  # NodeSet
    batch_id: str = "",
    dataset_id: Optional[str] = None,
) -> EpisodeContext:
    """
    Build EpisodeContext for a single episode.

    This helper function creates the complete context needed
    for all pipeline stages to process one episode.

    Phase 4-New-G: Helper for pipeline integration.
    """
    # For atomic episodes, skip FacetPoint extraction (single-sentence content
    # doesn't benefit from sub-facet decomposition, saves ~10s LLM call)
    effective_config = config
    if episode_memory_types.get(episode_id_str) == "atomic" and config.enable_facet_points:
        from dataclasses import replace
        effective_config = replace(config, enable_facet_points=False)

    return EpisodeContext(
        episode_id_str=episode_id_str,
        doc_summaries=doc_summaries,
        chunk_summaries=chunk_summaries,
        state=state,
        doc_title=doc_title,
        prev_title=state.title or "",
        prev_signature=state.signature or "",
        prev_summary=state.summary or "",
        doc_entity_cache=doc_entity_cache,
        episode_source_events=episode_source_events,
        original_event_routing_types=original_event_routing_types,
        routing_decisions=routing_decisions,
        episode_memory_types=episode_memory_types,
        config=effective_config,
        graph_engine=graph_engine,
        vector_engine=vector_engine,
        episodic_nodeset=episodic_nodeset,
        batch_id=batch_id,
        dataset_id=dataset_id,
    )


async def _process_single_episode_pipeline(
    ctx: EpisodeContext,
    existing_facets_lines: List[str],
) -> Tuple[Optional[Episode], List[Any], int, List[Any]]:
    """
    Process a single episode using the pipeline approach.

    This function orchestrates all the pipeline stages:
    1. Phase 0A: Three-way parallel (entity extraction, facet generation, matcher prep)
    2. Phase 0C: Entity creation
    3. Step 1: Time calculation + Facet preparation
    4. Step 2: Parallel entity description + FacetPoint extraction
    5. Step 3-5: Node and edge creation
    
    Args:
        ctx: Episode context with all inputs
        existing_facets_lines: Lines for prompt
    
    Returns:
        Tuple of:
        - Episode object (or None if processing failed)
        - List of procedural compile tasks
        - Number of same_entity_as edges added
        - List of EntityType objects to add to output

    Phase 4-New-G: Pipeline coordination function.
    """
    episode_id_str = ctx.episode_id_str

    # ========== Phase 0A: Three-way parallel ==========
    phase0a_result = await execute_phase0a(ctx)

    top_entity_names = phase0a_result.top_entity_names
    draft = phase0a_result.draft
    semantic_matcher = phase0a_result.semantic_matcher
    procedural_candidates = phase0a_result.procedural_candidates

    # Extract episode metadata from draft
    episode_name = draft.title
    episode_signature = draft.signature
    episode_summary = draft.summary

    # ========== Phase 0C: Entity creation ==========
    phase0c_result, same_entity_edges_pending = await execute_phase0c(
        top_entity_names=top_entity_names,
        episode_id_str=episode_id_str,
        episode_memory_types=ctx.episode_memory_types,
    )

    top_entities = phase0c_result.top_entities
    entity_map = phase0c_result.entity_map

    # Build compile tasks from procedural candidates
    procedural_compile_tasks: List[Any] = []
    if procedural_candidates and ctx.config.enable_procedural_routing:
        # Create nodeset for procedural memories
        procedural_nodeset = MemorySpace(
            id=generate_node_id("MemorySpace:Procedural"),
            name="Procedural",
        )

        async def _compile_one_candidate(entry: dict) -> Optional["Procedure"]:
            """Compile a single procedural candidate."""
            candidate = entry["candidate"]
            event_sentences = entry.get("event_sentences", "")
            source_episode_id = entry.get("episode_id", "")
            source_refs = [f"episode:{source_episode_id}"] if source_episode_id else None

            return await _compile_and_build_procedure(
                content=event_sentences,
                candidate=candidate,
                nodeset=procedural_nodeset,
                source_refs=source_refs,
            )

        # Create tasks (will be gathered by caller)
        import asyncio as _asyncio_pipeline

        for entry in procedural_candidates:
            task = _asyncio_pipeline.create_task(_compile_one_candidate(entry))
            procedural_compile_tasks.append(task)

        logger.info(
            f"[episodic.procedural_bridge] Started {len(procedural_candidates)} procedural compile tasks "
            f"(episode={episode_id_str[:20]}...)"
        )

    # ========== Step 1: Time + Facet preparation ==========
    step1_result = await execute_step1(
        episode_id_str=episode_id_str,
        episode_summary=episode_summary,
        doc_summaries=ctx.doc_summaries,
        state=ctx.state,
        top_entities=top_entities,
        draft=draft,
        semantic_matcher=semantic_matcher,
        episode_name=episode_name,
        episode_signature=episode_signature,
        max_new_facets_per_batch=ctx.config.max_new_facets_per_batch,
        enable_semantic_merge=ctx.config.enable_semantic_merge,
        evidence_chunks_per_facet=ctx.config.evidence_chunks_per_facet,
    )

    merged_time = step1_result.merged_time
    evidence_pairs = step1_result.evidence_pairs
    episode_name = step1_result.episode_name
    episode_signature = step1_result.episode_signature
    episode_summary = step1_result.episode_summary
    # Step 1 also returns updated facets
    updates = step1_result.updates
    existing_by_id = step1_result.existing_by_id

    # Calculate episode_time_fields from merged_time
    episode_time_fields = _extract_time_fields_from_episode(merged_time) if merged_time else {}

    # ========== Step 2: Parallel description + FacetPoint ==========
    step2_result = await execute_step2(
        top_entities=top_entities,
        entity_map=entity_map,
        chunk_summaries=ctx.chunk_summaries,
        updates=updates,
        existing_by_id=existing_by_id,
        evidence_pairs=evidence_pairs,
        state=ctx.state,
        graph_engine=ctx.graph_engine,
        enable_facet_points=ctx.config.enable_facet_points,
        facet_points_prompt_file=ctx.config.facet_points_prompt_file,
    )

    entity_context_map = step2_result.entity_context_map
    facet_points_cache = step2_result.facet_points_cache
    entity_type_cache = step2_result.entity_type_cache

    # ========== Step 3-5: Node and edge creation ==========

    # Step 3: Build has_facet_edges
    has_facet_edges = _build_has_facet_edges(
        updates=updates,
        existing_by_id=existing_by_id,
        facet_points_cache=facet_points_cache,
        evidence_pairs=evidence_pairs,
        merged_time=merged_time,
        episode_time_fields=episode_time_fields,
        episodic_nodeset=ctx.episodic_nodeset,
        enable_facet_points=ctx.config.enable_facet_points,
        dataset_id=ctx.dataset_id,
    )

    # Step 4: Build involves_entity_edges
    involves_entity_edges, involves_entities = _build_involves_entity_edges(
        entity_context_map=entity_context_map,
        entity_map=entity_map,
        top_entities=top_entities,
        episodic_nodeset=ctx.episodic_nodeset,
        max_entities_per_episode=ctx.config.max_entities_per_episode,
    )

    # Queue Facet-Entity edges
    _queue_facet_entity_edges(
        updates=updates,
        involves_entities=involves_entities,
        entity_map=entity_map,
        entity_context_map=entity_context_map,
    )

    # Collect same_entity_as edges
    same_entity_edges_count = _collect_same_entity_as_edges(
        same_entity_edges_pending=same_entity_edges_pending,
        involves_entities=involves_entities,
        entity_map=entity_map,
    )

    # Build includes_chunk_edges
    includes_chunk_edges = _build_includes_chunk_edges(ctx.doc_summaries)

    # Extract earliest created_at from source content for Episode timestamp
    # Priority: 1. Document.created_at (from API created_at parameter)
    #           2. ContentFragment.created_at (default: current system time)
    earliest_created_at = None
    for ts in ctx.doc_summaries:
        chunk = ts.made_from
        if chunk:
            # Priority 1: Try Document.created_at
            if hasattr(chunk, "is_part_of") and chunk.is_part_of:
                doc_created_at = getattr(chunk.is_part_of, "created_at", None)
                if doc_created_at is not None:
                    if earliest_created_at is None or doc_created_at < earliest_created_at:
                        earliest_created_at = doc_created_at
                    continue
            # Priority 2: Fall back to ContentFragment.created_at
            ch_created_at = getattr(chunk, "created_at", None)
            if ch_created_at is not None:
                if earliest_created_at is None or ch_created_at < earliest_created_at:
                    earliest_created_at = ch_created_at

    # Step 5: Create Episode object
    episode = _create_episode(
        episode_id_str=episode_id_str,
        episode_name=episode_name,
        episode_summary=episode_summary,
        episode_signature=episode_signature,
        has_facet_edges=has_facet_edges,
        involves_entity_edges=involves_entity_edges,
        includes_chunk_edges=includes_chunk_edges,
        episodic_nodeset=ctx.episodic_nodeset,
        episode_memory_types=ctx.episode_memory_types,
        routing_decisions=ctx.routing_decisions,
        state_exists=ctx.state.exists,
        state_memory_type=getattr(ctx.state, "memory_type", None),
        merged_time=merged_time,
        dataset_id=ctx.dataset_id,
        created_at=earliest_created_at,
    )

    # Collect EntityType objects for output
    concept_types = list(entity_type_cache.values()) if entity_type_cache else []

    return episode, procedural_compile_tasks, same_entity_edges_count, concept_types
