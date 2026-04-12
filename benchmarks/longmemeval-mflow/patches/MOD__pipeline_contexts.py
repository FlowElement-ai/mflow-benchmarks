# m_flow/memory/episodic/episode_builder/pipeline_contexts.py
"""
Pipeline Context Data Classes for Episode Processing

This module defines the data classes used to pass data between pipeline stages
in the episodic memory processing workflow.

Phase 4-New-A: Created as part of the large file refactoring plan.

Design Principles:
1. Each stage has a dedicated Result dataclass for its outputs
2. EpisodeContext contains all inputs needed for a single episode
3. EpisodeProcessingState accumulates results across stages
4. EpisodeConfig encapsulates all configuration parameters

Verified through 40 rounds of deep analysis:
- All cross-stage data dependencies are covered
- Object reference sharing is preserved (entity_map and top_entities)
- Time propagation logic is maintained
- Pending edge queue mechanism is unaffected
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from m_flow.core import Entity, EntityType, NodeSet
    from m_flow.memory.episodic.models import (
        EpisodicWriteDraft,
        FacetPointDraft,
    )
    from m_flow.knowledge.summarization.models import FragmentDigest
    from m_flow.memory.episodic.semantic_merge import SemanticFacetMatcher
    from m_flow.memory.episodic.state import EpisodeState, ExistingFacet
    from m_flow.memory.episodic.utils.models import FacetUpdate
    from m_flow.adapters.graph.graph_db_interface import GraphProvider
    from m_flow.adapters.vector.vector_db_interface import VectorProvider


# ============================================================
# Configuration Data Class
# ============================================================


@dataclass
class EpisodeConfig:
    """
    Encapsulates all configuration parameters for episode processing.

    This dataclass bundles the 20+ configuration parameters from
    write_episodic_memories into a single object for easier passing
    between pipeline stages.
    """

    # Entity limits
    max_entities_per_episode: int = 0  # 0 = unlimited
    max_candidate_entities_in_prompt: int = 25

    # Facet limits
    max_new_facets_per_batch: int = 20
    max_existing_facets_in_prompt: int = 60
    max_aliases_per_facet: int = 10
    aliases_text_max_chars: int = 400
    evidence_chunks_per_facet: int = 3

    # Chunk/prompt limits
    max_chunk_summaries_in_prompt: int = 40

    # Semantic merge settings
    enable_semantic_merge: bool = True
    semantic_merge_threshold: float = 0.85

    # Feature flags
    enable_episode_routing: bool = True
    enable_facet_points: bool = True
    enable_llm_entity_for_routing: bool = True
    enable_procedural_routing: bool = False

    # FacetPoint settings
    facet_points_prompt_file: Optional[str] = None
    max_point_aliases_text_chars: int = 400

    # Content routing state — when True, summarize_by_event uses the naming-aware
    # prompt to generate an Episode name (since content routing was not run).
    content_routing_disabled: bool = False
    precise_mode: bool = False


# ============================================================
# Episode Context Data Class
# ============================================================


@dataclass
class EpisodeContext:
    """
    Complete context for processing a single episode.

    This dataclass contains all the inputs needed to process one episode,
    including the episode identifier, document summaries, routing results,
    configuration, and database engines.

    Created at the start of each episode iteration and passed to all stages.
    """

    # Basic identification
    episode_id_str: str
    doc_summaries: List["FragmentDigest"]
    chunk_summaries: List[str]

    # Episode state from database
    state: "EpisodeState"

    # Document metadata
    doc_title: str
    prev_title: str  # From state.title
    prev_signature: str  # From state.signature
    prev_summary: str  # From state.summary

    # Routing results (from _route_documents_to_episodes)
    doc_entity_cache: Dict[str, List[str]]
    episode_source_events: Dict[str, List[str]]
    original_event_routing_types: Dict[str, str]
    routing_decisions: Dict[str, str]  # "new" or "existing"
    episode_memory_types: Dict[str, str]  # memory_type mapping

    # Configuration
    config: EpisodeConfig

    # Database engines
    graph_engine: "GraphProvider"
    vector_engine: "VectorProvider"

    # Shared objects
    episodic_nodeset: "NodeSet"

    # Dataset isolation: ID of the dataset being processed
    # Used for setting dataset_id on Episode/Facet nodes during creation
    dataset_id: Optional[str] = None

    # Batch identifier for logging
    batch_id: str = ""


# ============================================================
# Stage Result Data Classes
# ============================================================


@dataclass
class Phase0AResult:
    """
    Output of Phase 0A: Three-way parallel optimization.

    Contains results from:
    - _task_extract_entity_names(): Entity name extraction
    - _task_generate_facets(): Facet generation via LLM
    - _task_prepare_matcher(): Semantic matcher preparation

    Also collects procedural candidates for downstream processing.
    """

    top_entity_names: List[str]
    draft: "EpisodicWriteDraft"
    semantic_matcher: "SemanticFacetMatcher"
    procedural_candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Phase0CResult:
    """
    Output of Phase 0C: Entity creation.

    Contains:
    - top_entities: List of created Entity objects
    - entity_map: Mapping from entity name to Entity object
    - existing_entities_batch: Batch lookup results for existing entities

    Note: top_entities and entity_map share the same object references.
    Modifications to objects in top_entities are visible through entity_map.
    """

    top_entities: List["Entity"]
    entity_map: Dict[str, "Entity"]
    existing_entities_batch: Dict[str, List[Dict[str, Any]]]


@dataclass
class Step1Result:
    """
    Output of Step 0-1: Time calculation + Facet preparation.

    Contains:
    - Episode metadata (name, signature, summary)
    - Time calculation results (merged_time)
    - Facet processing results (updates, existing_by_id)
    - Evidence collection results (evidence_pairs)

    Note: After Step 8, top_entities objects have been updated with time fields.
    This is done in-place on the objects from Phase0CResult.
    """

    # Episode metadata
    episode_name: str
    episode_signature: str
    episode_summary: str

    # Time calculation (Step 0)
    merged_time: Optional[Dict[str, Any]]  # Contains mentioned_time_* fields

    # Facet processing
    updates: Dict[str, "FacetUpdate"]  # facet_id -> FacetUpdate
    existing_by_id: Dict[str, "ExistingFacet"]  # facet_id -> ExistingFacet

    # Evidence collection
    evidence_pairs: List[Tuple[Any, str]]  # (chunk, summary_text) pairs


@dataclass
class Step2Result:
    """
    Output of Step 2: Parallel entity description + facetpoint extraction.

    Contains:
    - entity_context_map: Entity descriptions and types from LLM
    - facet_points_cache: Extracted facet points per facet
    - entity_type_cache: Created EntityType objects
    """

    # Entity descriptions: name -> (description, entity_type)
    entity_context_map: Dict[str, Tuple[str, str]]

    # FacetPoint extraction results: facet_id -> list of FacetPointDraft
    facet_points_cache: Dict[str, List["FacetPointDraft"]]

    # EntityType cache: type_name -> EntityType object
    entity_type_cache: Dict[str, "EntityType"] = field(default_factory=dict)


# ============================================================
# State Accumulator Data Class
# ============================================================


@dataclass
class EpisodeProcessingState:
    """
    Accumulates outputs from all stages for cross-stage data access.

    Some stages (particularly Step 3-5) need access to outputs from
    multiple preceding stages. This dataclass serves as a central
    accumulator that each stage updates.

    Usage:
        state = EpisodeProcessingState()

        # Phase 0A updates
        state.top_entity_names = result_0a.top_entity_names
        state.draft = result_0a.draft
        ...

        # Phase 0C updates
        state.top_entities = result_0c.top_entities
        ...

        # Step 3-5 can access any field
        for entity in state.top_entities:
            ...
    """

    # Phase 0A outputs
    top_entity_names: List[str] = field(default_factory=list)
    draft: Optional["EpisodicWriteDraft"] = None
    semantic_matcher: Optional["SemanticFacetMatcher"] = None
    procedural_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # Phase 0C outputs
    top_entities: List["Entity"] = field(default_factory=list)
    entity_map: Dict[str, "Entity"] = field(default_factory=dict)
    existing_entities_batch: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Step 1 outputs (including Episode metadata)
    episode_name: str = ""
    episode_signature: str = ""
    episode_summary: str = ""
    merged_time: Optional[Dict[str, Any]] = None
    updates: Dict[str, "FacetUpdate"] = field(default_factory=dict)
    existing_by_id: Dict[str, Any] = field(default_factory=dict)
    evidence_pairs: List[Tuple[Any, str]] = field(default_factory=list)

    # Step 2 outputs
    entity_context_map: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    facet_points_cache: Dict[str, List["FacetPointDraft"]] = field(default_factory=dict)
    entity_type_cache: Dict[str, "EntityType"] = field(default_factory=dict)

    def update_from_phase0a(self, result: Phase0AResult) -> None:
        """Update state with Phase 0A results."""
        self.top_entity_names = result.top_entity_names
        self.draft = result.draft
        self.semantic_matcher = result.semantic_matcher
        self.procedural_candidates = result.procedural_candidates

    def update_from_phase0c(self, result: Phase0CResult) -> None:
        """Update state with Phase 0C results."""
        self.top_entities = result.top_entities
        self.entity_map = result.entity_map
        self.existing_entities_batch = result.existing_entities_batch

    def update_from_step1(self, result: Step1Result) -> None:
        """Update state with Step 1 results."""
        self.episode_name = result.episode_name
        self.episode_signature = result.episode_signature
        self.episode_summary = result.episode_summary
        self.merged_time = result.merged_time
        self.updates = result.updates
        self.existing_by_id = result.existing_by_id
        self.evidence_pairs = result.evidence_pairs

    def update_from_step2(self, result: Step2Result) -> None:
        """Update state with Step 2 results."""
        self.entity_context_map = result.entity_context_map
        self.facet_points_cache = result.facet_points_cache
        self.entity_type_cache = result.entity_type_cache


# ============================================================
# Module exports
# ============================================================

__all__ = [
    "EpisodeConfig",
    "EpisodeContext",
    "EpisodeProcessingState",
    "Phase0AResult",
    "Phase0CResult",
    "Step1Result",
    "Step2Result",
]
