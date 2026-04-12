import os
import asyncio
from typing import Literal, Set, Union, Optional
from uuid import UUID

from m_flow.config.config import get_memorize_config
from m_flow.shared.enums import ContentType
from m_flow.shared.logging_utils import get_logger
from m_flow.llm import get_max_chunk_tokens

from m_flow.pipeline import execute_workflow
from m_flow.pipeline.tasks import Stage
from m_flow.ingestion.chunking.TextChunker import TextChunker
from m_flow.auth.models import User
from m_flow.api.v1.exceptions.exceptions import ConcurrentMemorizeError

from m_flow.ingestion.documents import (
    detect_format,
    segment_documents,
)
from m_flow.storage import persist_memory_nodes
from m_flow.knowledge.summarization import compress_text
from m_flow.memory.episodic import (
    write_episodic_memories,
    write_same_entity_edges,
    write_facet_entity_edges,
    route_content_v2,  # Sentence-level routing
    run_episode_size_check,
    EpisodeSizeCheckConfig,
)
from m_flow.memory.procedural import write_procedural_memories
from m_flow.memory.procedural.write_procedural_from_episodic import (
    write_procedural_from_decisions,
)
from m_flow.pipeline.operations.execute_parallel import execute_parallel
from m_flow.pipeline.layers.pipeline_execution_mode import get_pipeline_executor


logger = get_logger("memorize")

# ============================================================================
# Memorize Concurrency Protection (P1)
# ============================================================================

# Active dataset IDs being processed (in-process memory, not cross-process)
_ACTIVE_DATASETS: Set[str] = set()
_ACTIVE_LOCK = asyncio.Lock()

# Conflict mode type
ConflictMode = Literal["ignore", "warn", "error"]


async def _check_and_register_datasets(
    dataset_ids: list[UUID],
    conflict_mode: ConflictMode,
) -> Set[str]:
    """
    Check and register active dataset IDs.

    Args:
        dataset_ids: List of dataset IDs to process.
        conflict_mode: Conflict handling mode.

    Returns:
        Set of registered dataset ID strings (for subsequent cleanup).

    Raises:
        ConcurrentMemorizeError: When conflict_mode="error" and conflict detected.
    """
    dataset_id_strs = {str(did) for did in dataset_ids}

    if not dataset_id_strs or conflict_mode == "ignore":
        return dataset_id_strs

    async with _ACTIVE_LOCK:
        conflicts = _ACTIVE_DATASETS & dataset_id_strs

        if conflicts:
            if conflict_mode == "error":
                raise ConcurrentMemorizeError(
                    f"Datasets {conflicts} are already being processed. "
                    f"Use conflict_mode='ignore' to proceed anyway."
                )
            elif conflict_mode == "warn":
                logger.warning(
                    f"[memorize] Concurrent processing detected for datasets: {conflicts}. "
                    f"This may cause data inconsistencies."
                )

        # Register as active
        _ACTIVE_DATASETS.update(dataset_id_strs)

    return dataset_id_strs


async def _unregister_datasets(dataset_id_strs: Set[str]) -> None:
    """Clean up active dataset registration."""
    if not dataset_id_strs:
        return

    async with _ACTIVE_LOCK:
        _ACTIVE_DATASETS.difference_update(dataset_id_strs)


async def _unified_episodic_procedural_write(
    memory_nodes: list,
    **kwargs,
) -> list:
    """
    Unified episodic + procedural write task.

    When both episodic and procedural are enabled:
    1. Write episodic memories with procedural routing enabled
    2. Episodic summarization includes procedural routing decision in same LLM call
    3. Collect procedural decisions
    4. Pass decisions to procedural pipeline for extraction

    This reduces LLM calls by combining episodic summary + procedural routing.
    """
    from m_flow.knowledge.summarization.models import FragmentDigest

    # Filter FragmentDigest from input
    summaries = [dp for dp in memory_nodes if isinstance(dp, FragmentDigest)]

    if not summaries:
        logger.info("[unified_write] No FragmentDigest found, returning input")
        return memory_nodes

    # Collect procedural decisions during episodic write
    procedural_decisions: list = []

    # Step 1: Write episodic memories with procedural routing enabled
    logger.info(f"[unified_write] Processing {len(summaries)} summaries with unified routing")
    episodic_result = await write_episodic_memories(
        summaries,
        enable_procedural_routing=True,
        procedural_decisions_out=procedural_decisions,
        **kwargs,
    )

    # Step 2: If procedural decisions found, process them
    if procedural_decisions:
        logger.info(f"[unified_write] Found {len(procedural_decisions)} procedural candidates")

        # Write procedural memories from decisions
        procedural_result = await write_procedural_from_decisions(
            decisions=procedural_decisions,
            **kwargs,
        )

        # Merge results
        return episodic_result + procedural_result
    else:
        logger.info("[unified_write] No procedural candidates detected")
        return episodic_result


async def memorize(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    chunker=TextChunker,
    chunk_size: int = None,
    chunks_per_batch: int = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    run_in_background: bool = False,
    incremental_loading: bool = True,
    enable_cache: bool = True,
    custom_prompt: Optional[str] = None,
    items_per_batch: int = 20,
    conflict_mode: ConflictMode = "warn",
    **kwargs,
):
    """
    Distil ingested documents into M-Flow's layered memory architecture.

    ``memorize`` is the central pipeline that reads raw data previously
    registered via :func:`m_flow.add`, breaks it into semantic fragments,
    and writes structured memory layers:

    * **Episodic layer** — Episode nodes anchored by summaries, linked to
      Facet and Entity nodes via rich semantic edges (the *Cone Graph*).
    * **Procedural layer** (opt-in) — reusable process/preference knowledge
      compiled from episodic content by a secondary LLM pass.

    The pipeline proceeds through these stages:

    1. Classify documents by type.
    2. Chunk text into semantically coherent fragments.
    3. (Optional) Sentence-level content routing — tag each sentence as
       episodic, atomic, or procedural before summarisation.
    4. Summarise fragments into ``FragmentDigest`` objects.
    5. Write episodic memories (Episodes, Facets, Entities).
    6. Persist all MemoryNodes to graph + vector stores.
    7. Post-process: ``same_entity_as`` edges, ``facet_entity`` edges.

    Parameters
    ----------
    datasets : str | list[str] | list[UUID] | None
        Which datasets to process.  ``None`` means all datasets owned by *user*.
    user : User | None
        Caller identity; falls back to the default seed user.
    chunker : type
        Chunking strategy class (default ``TextChunker``).
    chunk_size : int | None
        Max tokens per chunk.  Auto-selected from the LLM context window when omitted.
    chunks_per_batch : int | None
        How many chunks a single pipeline task processes in one go.
    vector_db_config, graph_db_config : dict | None
        Override the global adapter configuration for this run.
    run_in_background : bool
        When True, schedule the pipeline asynchronously and return immediately.
    enable_cache : bool
        Skip re-processing datasets that have already been memorised.
    custom_prompt : str | None
        Replace the default extraction prompt for the LLM knowledge-graph step.
    items_per_batch : int
        Number of data items processed concurrently per batch (default 20).
    conflict_mode : ``"warn"`` | ``"error"`` | ``"ignore"``
        How to handle concurrent memorize calls on the same dataset.
        ``"error"`` raises :class:`ConcurrentMemorizeError`.

    Returns
    -------
    dict | list[RunEvent]
        In blocking mode a mapping of ``dataset_id -> RunEvent``; in background
        mode a list of ``RunEvent`` objects for progress tracking.

    Raises
    ------
    ConcurrentMemorizeError
        When *conflict_mode* is ``"error"`` and the dataset is already being processed.
    """
    # --- Concurrency Protection (P1) ---
    # Parse dataset IDs (outside lock to avoid I/O inside lock)
    from m_flow.auth.methods import get_seed_user
    from m_flow.data.methods import get_dataset_ids

    active_user = user
    if active_user is None:
        try:
            active_user = await get_seed_user()
        except Exception as e:
            # If user retrieval fails, skip conflict detection
            logger.debug("Failed to get default user for conflict check: %s", e)
            active_user = None

    # Normalize datasets to list
    ds_list: list = []
    if datasets is not None:
        ds_list = [datasets] if isinstance(datasets, str) else list(datasets)

    # Get dataset IDs
    dataset_ids: list[UUID] = []
    if ds_list and active_user is not None:
        try:
            dataset_ids = await get_dataset_ids(ds_list, active_user)
        except Exception as e:
            logger.debug(f"[memorize] Failed to resolve dataset IDs for conflict check: {e}")
            # Continue processing, skip conflict detection

    # Register and check conflicts
    registered_ids: Set[str] = set()
    if dataset_ids:
        registered_ids = await _check_and_register_datasets(dataset_ids, conflict_mode)

    try:
        # --- Original logic ---
        tasks = await get_default_tasks(
            user=user,
            chunker=chunker,
            chunk_size=chunk_size,
            custom_prompt=custom_prompt,
            chunks_per_batch=chunks_per_batch,
            **kwargs,
        )

        # Obtain an executor that either runs the pipeline in the background or blocks until completion
        pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

        # Run the execute_workflow in the background or blocking based on executor
        from m_flow.pipeline.operations.pipeline import WorkflowConfig
        result = await pipeline_executor_func(
            pipeline=execute_workflow,
            tasks=tasks,
            user=user,
            datasets=datasets,
            name="memorize_pipeline",
            config=WorkflowConfig(
                vector_db=vector_db_config,
                graph_db=graph_db_config,
                cache=enable_cache,
                incremental=incremental_loading,
                batch_size=items_per_batch,
            ),
        )

        # Episode Size Check: Post-memorize maintenance
        # Controlled by MFLOW_EPISODE_SIZE_CHECK_AUTO environment variable (default: true)
        episode_size_check_auto = os.getenv(
            "MFLOW_EPISODE_SIZE_CHECK_AUTO", "true"
        ).lower() not in (
            "0",
            "false",
            "no",
            "n",
            "off",
        )

        if episode_size_check_auto and not run_in_background:
            # Only run in blocking mode - background mode returns immediately
            logger.info("[memorize] Running automatic Episode Size Check")
            try:
                check_stats = await run_episode_size_check(EpisodeSizeCheckConfig())
                logger.info(
                    f"[memorize] Episode Size Check: checked={check_stats['checked']}, "
                    f"split={check_stats['split']}, adapted={check_stats['adapted']}"
                )
            except Exception as e:
                logger.warning(f"[memorize] Episode Size Check failed: {e}")

        # Force Kuzu checkpoint to persist WAL data to disk
        # This prevents data loss on abnormal shutdown (e.g., kill -9)
        if not run_in_background:
            try:
                from m_flow.adapters.graph.get_graph_adapter import get_graph_provider
                graph_engine = await get_graph_provider()
                if hasattr(graph_engine, 'checkpoint'):
                    await graph_engine.checkpoint()
                    logger.info("[memorize] Graph database checkpoint completed")
            except Exception as e:
                logger.warning(f"[memorize] Graph checkpoint failed: {e}")

        return result

    finally:
        # --- Clean up concurrent registration (P1) ---
        if registered_ids:
            await _unregister_datasets(registered_ids)


async def get_default_tasks(
    user: User = None,
    chunker=TextChunker,
    chunk_size: int = None,
    custom_prompt: Optional[str] = None,
    chunks_per_batch: int = 100,
    **kwargs,
) -> list[Stage]:
    if chunks_per_batch is None:
        chunks_per_batch = 100

    memorize_config = get_memorize_config()
    embed_triplets = memorize_config.triplet_embedding

    # ---------------------------------------------------------------------------
    # Feature Toggles (Priority: kwargs > environment variable > default)
    # ---------------------------------------------------------------------------
    
    def _env_bool(env_var: str, default: bool) -> bool:
        """Parse environment variable as boolean."""
        val = os.getenv(env_var, str(default).lower()).lower()
        return val in ("1", "true", "yes", "y", "on")

    # Extract feature toggles from kwargs (API-level override)
    enable_episodic_override = kwargs.pop("enable_episodic", None)
    enable_procedural_override = kwargs.pop("enable_procedural", None)
    enable_content_routing_override = kwargs.pop("enable_content_routing", None)
    
    # Extract content_type for sentence splitting strategy
    content_type = kwargs.pop("content_type", ContentType.TEXT)

    # Precise mode: two-step summarization (JSON section routing + per-section compression)
    precise_mode = kwargs.pop("precise_mode", _env_bool("MFLOW_PRECISE_MODE", False))
    if precise_mode:
        logger.info("[memorize] Precise summarization mode enabled")

    # Episodic memory layer: kwargs > MFLOW_EPISODIC_ENABLED > true
    if enable_episodic_override is not None:
        episodic_enabled = enable_episodic_override
    else:
        episodic_enabled = _env_bool("MFLOW_EPISODIC_ENABLED", True)

    # Procedural memory layer: kwargs > MFLOW_PROCEDURAL_ENABLED > false
    if enable_procedural_override is not None:
        procedural_enabled = enable_procedural_override
    else:
        procedural_enabled = _env_bool("MFLOW_PROCEDURAL_ENABLED", False)

    # Content Routing: kwargs > MFLOW_CONTENT_ROUTING > true
    if enable_content_routing_override is not None:
        content_routing_enabled = enable_content_routing_override
    else:
        content_routing_enabled = _env_bool("MFLOW_CONTENT_ROUTING", True)

    # Sentence-level routing is the current implementation
    sentence_routing_enabled = content_routing_enabled

    if procedural_enabled:
        logger.info("[memorize] Procedural memory enabled")

    # Sentence-Level Content Routing Pipeline
    if sentence_routing_enabled and episodic_enabled:
        logger.info("[memorize] Sentence-level content routing enabled")

        # Build Memory write task (supports Procedural unified routing)
        def _build_memory_task():
            """
            Build Memory write task for Sentence-Level Content Routing Pipeline.

            When procedural is enabled:
            - Episodic summarization includes procedural routing decision in same LLM call
            - Procedural decisions are collected and passed to procedural pipeline
            - This reduces LLM calls by combining episodic summary + procedural routing
            """
            if procedural_enabled:
                logger.info(
                    "[memorize] Sentence routing + Procedural: unified routing in episodic summarization"
                )
                # Use unified episodic+procedural task
                return Stage(
                    _unified_episodic_procedural_write,
                    task_config={"batch_size": chunks_per_batch},
                )
            else:
                return Stage(write_episodic_memories, precise_mode=precise_mode, task_config={"batch_size": chunks_per_batch})

        memory_task = _build_memory_task()

        # Sentence-level Pipeline: route_content_v2 before compress_text
        # Works at ContentFragment level, FragmentDigest is not needed
        default_tasks = [
            Stage(detect_format),
            Stage(
                segment_documents,
                max_chunk_size=chunk_size or get_max_chunk_tokens(),
                chunker=chunker,
            ),
            # Step 1: Sentence-level routing (BEFORE summarize)
            # Adds sentence_classifications to chunk.metadata
            Stage(
                route_content_v2,
                content_type=content_type,  # Explicit content type for sentence splitting
                task_config={"batch_size": chunks_per_batch},
            ),
            # Step 2: Summarize (unaffected by sentence routing)
            Stage(
                compress_text,
                task_config={"batch_size": chunks_per_batch},
            ),
            # Memory processing (reads sentence classification metadata)
            # Processes BOTH episodic AND atomic content as Episodes
            memory_task,
            # Step 5: Persist all MemoryNodes
            Stage(
                persist_memory_nodes,
                embed_triplets=embed_triplets,
                task_config={"batch_size": chunks_per_batch},
            ),
            # Post-processing
            Stage(write_same_entity_edges, task_config={"batch_size": chunks_per_batch}),
            Stage(write_facet_entity_edges, task_config={"batch_size": chunks_per_batch}),
        ]

        return default_tasks

    else:
        # Original Pipeline
        # Build Memory task (Episodic + Procedural can execute in parallel)
        def _build_memory_task():
            """Build Memory write task, supports parallel execution of Episodic and Procedural"""
            if episodic_enabled and procedural_enabled:
                # Execute Episodic and Procedural in parallel
                logger.info("[memorize] Parallel execution: Episodic + Procedural")
                return execute_parallel(
                    [
                        Stage(write_episodic_memories, precise_mode=precise_mode, task_config={"batch_size": chunks_per_batch}),
                        Stage(
                            write_procedural_memories, task_config={"batch_size": chunks_per_batch}
                        ),
                    ],
                    merge_results=True,
                    deduplicate=True,
                )
            elif episodic_enabled:
                return Stage(write_episodic_memories, precise_mode=precise_mode, task_config={"batch_size": chunks_per_batch})
            elif procedural_enabled:
                return Stage(write_procedural_memories, task_config={"batch_size": chunks_per_batch})
            else:
                return None

        memory_task = _build_memory_task()

        default_tasks = [
            Stage(detect_format),
            Stage(
                segment_documents,
                max_chunk_size=chunk_size or get_max_chunk_tokens(),
                chunker=chunker,
            ),
            Stage(
                compress_text,
                task_config={"batch_size": chunks_per_batch},
            ),
            # Memory layer: Episodic + Procedural (can execute in parallel)
            # Inserted after compress_text, before persist_memory_nodes
            *([memory_task] if memory_task is not None else []),
            Stage(
                persist_memory_nodes,
                embed_triplets=embed_triplets,
                task_config={"batch_size": chunks_per_batch},
            ),
            # Post-processing: write same_entity_as edges (after Entity nodes are in graph)
            # Only execute when Episodic is enabled
            *(
                [
                    Stage(write_same_entity_edges, task_config={"batch_size": chunks_per_batch}),
                    Stage(write_facet_entity_edges, task_config={"batch_size": chunks_per_batch}),
                ]
                if episodic_enabled
                else []
            ),
        ]

    return default_tasks
