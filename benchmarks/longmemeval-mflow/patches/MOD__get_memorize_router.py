import os
import asyncio
from uuid import UUID
from pydantic import Field
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION

from m_flow.api.DTO import InDTO
from m_flow.pipeline.methods import get_pipeline_run
from m_flow.auth.models import User
from m_flow.auth.methods import get_authenticated_user
from m_flow.auth.get_user_db import get_user_db_context
from m_flow.knowledge.graph_ops.methods import get_formatted_graph_data
from m_flow.auth.get_user_manager import get_user_manager_context
from m_flow.adapters.relational import get_db_adapter
from m_flow.auth.authentication.default.default_jwt_strategy import DefaultJWTStrategy
from m_flow.auth.security_check import get_secret_with_production_check
from m_flow.pipeline.models.RunEvent import (
    RunAlreadyCompleted,
    RunCompleted,
    RunEvent,
    RunFailed,
)
from m_flow.pipeline.queues.workflow_run_info_queues import (
    get_from_queue,
    initialize_queue,
    remove_queue,
)
from m_flow.shared.logging_utils import get_logger
from m_flow.shared.utils import send_telemetry
from m_flow import __version__ as m_flow_version
from m_flow.ingestion.chunking.TextChunker import TextChunker
from m_flow.ingestion.chunking.LangchainChunker import LangchainChunker

logger = get_logger("api.memorize")

CHUNKER_MAP = {
    "TextChunker": TextChunker,
    "LangchainChunker": LangchainChunker,
}


class MemorizePayloadDTO(InDTO):
    """
    Memorize request payload.
    
    Supports all configuration options for knowledge graph construction,
    including feature toggles, chunking settings, and processing parameters.
    """
    
    # Dataset selection
    datasets: Optional[List[str]] = Field(default=None)
    dataset_ids: Optional[List[UUID]] = Field(default=None, examples=[[]])
    
    # Execution mode
    run_in_background: Optional[bool] = Field(
        default=False,
        description="If true, process asynchronously and return immediately"
    )
    
    # Custom prompt
    custom_prompt: Optional[str] = Field(
        default="",
        description="Custom prompt for entity extraction and graph generation"
    )
    
    # Chunking configuration
    chunk_size: Optional[int] = Field(
        default=None,
        description="Maximum tokens per chunk. Auto-calculated based on LLM if None."
    )
    chunker: Optional[str] = Field(
        default="TextChunker",
        description="Text chunking strategy: 'TextChunker' (paragraph-based) or 'LangchainChunker' (recursive with overlap)"
    )
    chunks_per_batch: Optional[int] = Field(
        default=100,
        description="Number of chunks to process in a single batch"
    )
    items_per_batch: Optional[int] = Field(
        default=20,
        description="Number of data items (files/texts) to process per batch"
    )
    
    # Processing behavior
    incremental_loading: Optional[bool] = Field(
        default=True,
        description="Skip already processed files/content"
    )
    conflict_mode: Optional[str] = Field(
        default="warn",
        description="Concurrent conflict handling: 'warn', 'error', or 'ignore'"
    )
    
    # ========================================================================
    # Feature toggles (override environment variables)
    # ========================================================================
    enable_episode_routing: Optional[bool] = Field(
        default=None,
        description="Enable cross-batch incremental update routing. "
                    "When enabled, new content can be merged into existing episodes. "
                    "Set to false for isolated batch processing. "
                    "Default: uses MFLOW_EPISODIC_ENABLE_ROUTING env var (true)."
    )
    enable_content_routing: Optional[bool] = Field(
        default=None,
        description="Enable sentence-level content classification. "
                    "Default: uses MFLOW_CONTENT_ROUTING env var (true)."
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Content type declaration. Required when enable_content_routing=True. "
                    "Values: 'text' for articles/documents, 'dialog' for conversations/chat logs."
    )
    enable_procedural: Optional[bool] = Field(
        default=None,
        description="Enable procedural memory extraction. "
                    "Default: uses MFLOW_PROCEDURAL_ENABLED env var (true)."
    )
    enable_semantic_merge: Optional[bool] = Field(
        default=None,
        description="Enable semantic merge for similar facets. "
                    "Default: uses MFLOW_EPISODIC_ENABLE_SEMANTIC_MERGE env var (false)."
    )
    enable_facet_points: Optional[bool] = Field(
        default=None,
        description="Enable FacetPoint three-layer structure. "
                    "Default: uses MFLOW_EPISODIC_ENABLE_FACET_POINTS env var (true)."
    )
    extract_relationships: Optional[bool] = Field(
        default=None,
        description="Enable relationship edge extraction between entities. "
                    "Default: true."
    )
    precise_mode: Optional[bool] = Field(
        default=None,
        description="Enable precise summarization: preserves all factual details (dates, numbers, names) "
                    "with lower compression ratio — RAG context will be longer but more accurate. "
                    "Default: uses MFLOW_PRECISE_MODE env var (false)."
    )


def get_memorize_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def memorize(payload: MemorizePayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Transform datasets into structured knowledge graphs through cognitive processing.

        This endpoint is the core of Mflow's intelligence layer, responsible for converting
        raw text, documents, and data added through the add endpoint into semantic knowledge graphs.
        It performs deep analysis to extract entities, relationships, and insights from ingested content.

        ## Processing Pipeline
        1. Document classification and permission validation
        2. Text chunking and semantic segmentation
        3. Entity extraction using LLM-powered analysis
        4. Relationship detection and graph construction
        5. Vector embeddings generation for semantic search
        6. Content summarization and indexing

        ## Request Parameters
        - **datasets** (Optional[List[str]]): List of dataset names to process. Dataset names are resolved to datasets owned by the authenticated user.
        - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow processing of datasets not owned by the user (if permitted).
        - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to False (blocking).
        - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If provided, this prompt will be used instead of the default prompts for knowledge graph extraction.
        - **chunk_size** (Optional[int]): Maximum tokens per chunk. Auto-calculated based on LLM context window if None.
        - **chunker** (Optional[str]): Text chunking strategy - 'TextChunker' (paragraph-based, default) or 'LangchainChunker' (recursive with overlap).
        - **chunks_per_batch** (Optional[int]): Number of chunks to process in a single batch. Defaults to 100.
        - **items_per_batch** (Optional[int]): Number of data items (files/texts) to process per batch. Defaults to 20.
        - **incremental_loading** (Optional[bool]): Skip already processed files/content. Defaults to True.
        - **conflict_mode** (Optional[str]): Concurrent conflict handling - 'warn' (default), 'error', or 'ignore'.

        ## Response
        - **Blocking execution**: Complete pipeline run information with entity counts, processing duration, and success/failure status
        - **Background execution**: Pipeline run metadata including workflow_run_id for status monitoring via WebSocket subscription

        ## Error Codes
        - **400 Bad Request**: When neither datasets nor dataset_ids are provided, or when specified datasets don't exist
        - **409 Conflict**: When processing fails due to system errors, missing LLM API keys, database connection failures, or corrupted content

        ## Example Request
        ```json
        {
            "datasets": ["research_papers", "documentation"],
            "run_in_background": false,
            "custom_prompt": "Extract entities focusing on technical concepts and their relationships.",
            "chunk_size": 2048,
            "chunker": "TextChunker",
            "chunks_per_batch": 100,
            "incremental_loading": true
        }
        ```

        ## Notes
        To memorize data in datasets not owned by the user and for which the current user has write permission,
        the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

        ## Next Steps
        After successful processing, use the search endpoints to query the generated knowledge graph for insights, relationships, and semantic search.
        """
        send_telemetry(
            "Memorize API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/memorize",
                "m_flow_version": m_flow_version,
            },
        )

        if not payload.datasets and not payload.dataset_ids:
            return JSONResponse(
                status_code=400, content={"error": "No datasets or dataset_ids provided"}
            )

        from m_flow.api.v1.memorize import memorize as m_flow_memorize

        try:
            datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets

            # Resolve chunker class from string
            chunker_class = CHUNKER_MAP.get(payload.chunker, TextChunker)

            # Build kwargs for feature toggles (only pass if explicitly set)
            kwargs = {}
            if payload.enable_episode_routing is not None:
                kwargs["enable_episode_routing"] = payload.enable_episode_routing
            if payload.enable_content_routing is not None:
                kwargs["enable_content_routing"] = payload.enable_content_routing
            if payload.content_type is not None:
                from m_flow.shared.enums import ContentType
                kwargs["content_type"] = ContentType(payload.content_type)
            if payload.enable_procedural is not None:
                kwargs["enable_procedural"] = payload.enable_procedural
            if payload.enable_semantic_merge is not None:
                kwargs["enable_semantic_merge"] = payload.enable_semantic_merge
            if payload.enable_facet_points is not None:
                kwargs["enable_facet_points"] = payload.enable_facet_points
            if payload.extract_relationships is not None:
                kwargs["extract_relationships"] = payload.extract_relationships
            if payload.precise_mode is not None:
                kwargs["precise_mode"] = payload.precise_mode

            memorize_run = await m_flow_memorize(
                datasets,
                user,
                run_in_background=payload.run_in_background,
                custom_prompt=payload.custom_prompt,
                chunk_size=payload.chunk_size,
                chunker=chunker_class,
                chunks_per_batch=payload.chunks_per_batch,
                items_per_batch=payload.items_per_batch,
                incremental_loading=payload.incremental_loading,
                conflict_mode=payload.conflict_mode,
                **kwargs,
            )

            # If any memorize run errored return JSONResponse with proper error status code
            if any(isinstance(v, RunFailed) for v in memorize_run.values()):
                return JSONResponse(status_code=420, content=jsonable_encoder(memorize_run))
            return memorize_run
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.websocket("/subscribe/{workflow_run_id}")
    async def subscribe_to_memorize_info(websocket: WebSocket, workflow_run_id: str):
        await websocket.accept()

        # Support both query parameter token and cookie
        access_token = (
            websocket.query_params.get("token")
            or websocket.cookies.get(os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"))
        )

        if not access_token:
            logger.error("WebSocket: no token provided")
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="No token")
            return

        try:
            secret = get_secret_with_production_check(
                "FASTAPI_USERS_JWT_SECRET", "super_secret", "JWT WebSocket authentication"
            )

            strategy = DefaultJWTStrategy(secret, lifetime_seconds=3600)

            db_engine = get_db_adapter()

            async with db_engine.get_async_session() as session:
                async with get_user_db_context(session) as user_db:
                    async with get_user_manager_context(user_db) as user_manager:
                        user = await strategy.read_token(access_token, user_manager)
        except Exception as error:
            logger.error(f"Authentication failed: {str(error)}")
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return

        workflow_run_id = UUID(workflow_run_id)

        pipeline_run = await get_pipeline_run(workflow_run_id)

        initialize_queue(workflow_run_id)

        while True:
            workflow_run_info = get_from_queue(workflow_run_id)

            if not workflow_run_info:
                await asyncio.sleep(2)
                continue

            if not isinstance(workflow_run_info, RunEvent):
                continue

            try:
                await websocket.send_json(
                    {
                        "workflow_run_id": str(workflow_run_info.workflow_run_id),
                        "status": workflow_run_info.status,
                        "payload": await get_formatted_graph_data(pipeline_run.dataset_id, user),
                    }
                )

                # Close connection for terminal states
                if isinstance(workflow_run_info, RunCompleted):
                    remove_queue(workflow_run_id)
                    await websocket.close(code=WS_1000_NORMAL_CLOSURE)
                    break
                elif isinstance(workflow_run_info, RunAlreadyCompleted):
                    # Pipeline was already completed (cached result)
                    remove_queue(workflow_run_id)
                    await websocket.close(code=WS_1000_NORMAL_CLOSURE)
                    break
                elif isinstance(workflow_run_info, RunFailed):
                    # Pipeline errored - close with error indicator
                    remove_queue(workflow_run_id)
                    logger.warning(f"Pipeline {workflow_run_id} errored, closing WebSocket")
                    await websocket.close(code=WS_1000_NORMAL_CLOSURE, reason="Pipeline errored")
                    break
            except WebSocketDisconnect:
                remove_queue(workflow_run_id)
                break

    return router
