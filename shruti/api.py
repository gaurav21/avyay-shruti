"""
FastAPI application for ŚRUTI package.
Provides REST API endpoints for transcription, extraction, ingestion, and querying.
"""

import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_config, validate_config, setup_directories
from .middleware import (
    RateLimitMiddleware,
    ALLOWED_ORIGINS,
    TIER_LIMITS,
    Tier,
)
from .models import (
    TranscribeRequest, TranscribeResponse,
    ExtractRequest, ExtractResponse,
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    BatchQueryRequest, BatchQueryResponse,
    DatabaseStatsResponse, HealthResponse,
    ErrorResponse, ConfigInfo,
    RelatedTopicsRequest, RelatedTopicsResponse,
    MetadataSearchRequest, MetadataSearchResponse,
    # V2 models
    SegmentationResponse, TopicModel, ChapterModel,
    KnowledgeGraphResponse, GraphEntityModel,
    MultiModalRequest, MultiModalResponse,
    # V3 summarization models
    SummarizeRequest, SummarizeResponse,
    InsightModel, ActionItemModel, SentimentSegmentModel,
    EnhancedChapterModel, QualityReportModel,
    InsightSearchRequest, InsightSearchResponse,
)
from .transcribe import transcribe_url
from .extract import extract_knowledge, extract_knowledge_batch
from .embeddings import add_video_to_vectorstore, get_database_stats
from .query import (
    query_knowledge, batch_query_knowledge, 
    get_related_topics, search_by_metadata
)
from .segments import segment_content, generate_youtube_chapters
from .summarize import summarize_video, SummarizationConfig, ChapterDetectionConfig
from .exports import (
    export_summary_markdown, export_summary_json,
    export_chapter_timestamps, save_insights_to_db, search_insights,
)
from .knowledge_graph import (
    extract_entities, extract_relationships,
    build_knowledge_graph, load_knowledge_graph,
    save_knowledge_graph, get_graph_statistics, query_graph,
)

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ŚRUTI (श्रुति) API",
    description="Multilingual YouTube Knowledge Extractor - Extract, process, and query knowledge from YouTube videos",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting middleware (applied first, before CORS)
app.add_middleware(RateLimitMiddleware)

# CORS — restricted to Avyay domains (not wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "X-API-Key",
        "X-ReCAPTCHA-Token",
        "Content-Type",
        "Accept",
    ],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-RateLimit-Tier",
        "Retry-After",
    ],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            error_type=type(exc).__name__
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        setup_directories()
        # Don't validate GROQ_API_KEY on startup — it's only needed for
        # transcription/extraction endpoints, not health checks.
        # This allows the container to start and serve /health even
        # if the API key isn't configured yet.
        try:
            validate_config()
            logger.info("ŚRUTI API server started successfully (config valid)")
        except ValueError as e:
            logger.warning(f"Config validation warning (non-fatal): {e}")
            logger.info("ŚRUTI API server started (some features may be unavailable)")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "ŚRUTI (श्रुति) API",
        "version": __version__,
        "description": "Multilingual YouTube Knowledge Extractor",
        "docs": "/docs",
        "health": "/health",
        "rate_limits": {
            tier.value: f"{limit} req/hour"
            for tier, limit in TIER_LIMITS.items()
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check configuration
        config_valid = True
        try:
            validate_config()
        except Exception:
            config_valid = False
        
        # Check database accessibility
        database_accessible = True
        try:
            get_database_stats()
        except Exception:
            database_accessible = False
        
        status = "healthy" if config_valid and database_accessible else "unhealthy"
        
        return HealthResponse(
            status=status,
            version=__version__,
            config_valid=config_valid,
            database_accessible=database_accessible
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.get("/config", response_model=ConfigInfo)
async def get_config_info():
    """Get current configuration information."""
    try:
        config = get_config()
        return ConfigInfo(
            embedding_model=config.EMBEDDING_MODEL,
            llm_model=config.LLM_MODEL,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            chroma_persist_dir=config.CHROMA_PERSIST_DIR
        )
    except Exception as e:
        logger.error(f"Failed to get config info: {e}")
        raise HTTPException(status_code=500, detail=f"Config error: {e}")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_video(request: TranscribeRequest):
    """Transcribe a YouTube video."""
    try:
        logger.info(f"Transcribing video: {request.url}")
        result = transcribe_url(request.url)
        
        return TranscribeResponse(
            video_id=result["video_id"],
            title=result["title"],
            transcript=result["transcript"],
            language=result["language"],
            source=result["source"],
            duration=result.get("duration"),
            url=result["url"],
            segments=result.get("segments")
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")


@app.post("/extract", response_model=ExtractResponse)
async def extract_knowledge_from_video(request: ExtractRequest):
    """Extract structured knowledge from a YouTube video."""
    try:
        logger.info(f"Extracting knowledge from: {request.url}")
        
        # Transcribe the video
        transcript_result = transcribe_url(request.url)
        
        # Extract knowledge
        knowledge = extract_knowledge(
            transcript=transcript_result["transcript"],
            title=transcript_result["title"]
        )
        
        return ExtractResponse(
            video_id=transcript_result["video_id"],
            title=transcript_result["title"],
            knowledge=knowledge,
            transcript_source=transcript_result["source"],
            language=transcript_result["language"]
        )
        
    except Exception as e:
        logger.error(f"Knowledge extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Knowledge extraction failed: {e}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_video(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a video into the knowledge base (transcribe + chunk + embed + store)."""
    try:
        logger.info(f"Ingesting video: {request.url}")
        
        # Transcribe the video
        transcript_result = transcribe_url(request.url)
        
        # Prepare metadata
        metadata = {
            "video_id": transcript_result["video_id"],
            "title": transcript_result["title"],
            "language": transcript_result["language"],
            "source": transcript_result["source"],
            "duration": transcript_result.get("duration", 0),
            "url": transcript_result["url"],
        }
        
        # Add to vector store
        chunks_stored = add_video_to_vectorstore(
            transcript=transcript_result["transcript"],
            metadata=metadata,
            persist_dir=request.persist_dir
        )
        
        config = get_config()
        persist_dir = request.persist_dir or str(config.get_chroma_persist_path())
        
        return IngestResponse(
            video_id=transcript_result["video_id"],
            title=transcript_result["title"],
            chunks_stored=chunks_stored,
            language=transcript_result["language"],
            transcript_source=transcript_result["source"],
            persist_dir=persist_dir
        )
        
    except Exception as e:
        logger.error(f"Video ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=f"Video ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base using RAG."""
    try:
        logger.info(f"Querying knowledge base: {request.query[:100]}...")
        
        result = query_knowledge(
            question=request.query,
            top_k=request.top_k,
            persist_dir=request.persist_dir,
            include_metadata=request.include_metadata,
            temperature=request.temperature
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            chunks_found=result["chunks_found"],
            avg_similarity=result["avg_similarity"],
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=400, detail=f"Query failed: {e}")


@app.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query_knowledge_base(request: BatchQueryRequest):
    """Process multiple queries in batch."""
    try:
        logger.info(f"Processing batch of {len(request.queries)} queries")
        
        results = batch_query_knowledge(
            questions=request.queries,
            top_k=request.top_k,
            persist_dir=request.persist_dir
        )
        
        # Convert to QueryResponse objects
        query_responses = []
        successful_count = 0
        
        for result in results:
            if "error" not in result:
                successful_count += 1
            
            query_responses.append(QueryResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                confidence=result.get("confidence", "error"),
                chunks_found=result.get("chunks_found", 0),
                avg_similarity=result.get("avg_similarity", 0.0),
                query=result["question"]
            ))
        
        return BatchQueryResponse(
            results=query_responses,
            total_queries=len(request.queries),
            successful_queries=successful_count
        )
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(status_code=400, detail=f"Batch query failed: {e}")


@app.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_statistics():
    """Get knowledge base statistics."""
    try:
        stats = get_database_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return DatabaseStatsResponse(
            total_chunks=stats["total_chunks"],
            unique_videos=stats["unique_videos"],
            languages=stats["languages"],
            persist_dir=stats["persist_dir"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")


@app.post("/related-topics", response_model=RelatedTopicsResponse)
async def find_related_topics(request: RelatedTopicsRequest):
    """Find topics related to the given topic."""
    try:
        related_topics = get_related_topics(
            topic=request.topic,
            top_k=request.top_k,
            persist_dir=request.persist_dir
        )
        
        return RelatedTopicsResponse(
            topic=request.topic,
            related_topics=related_topics
        )
        
    except Exception as e:
        logger.error(f"Failed to find related topics: {e}")
        raise HTTPException(status_code=400, detail=f"Related topics search failed: {e}")


@app.post("/search/metadata", response_model=MetadataSearchResponse)
async def search_by_metadata_filters(request: MetadataSearchRequest):
    """Search chunks by metadata filters."""
    try:
        results = search_by_metadata(
            metadata_filter=request.filters,
            limit=request.limit,
            persist_dir=request.persist_dir
        )
        
        return MetadataSearchResponse(
            results=results,
            total_found=len(results),
            filters_applied=request.filters
        )
        
    except Exception as e:
        logger.error(f"Metadata search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Metadata search failed: {e}")


# ===== V2.0 Multi-Modal Endpoints =====

@app.post("/v2/segment")
async def segment_video_content(request: ExtractRequest):
    """Segment a video into topics and auto-detect chapters."""
    try:
        logger.info(f"Segmenting content from: {request.url}")

        # Transcribe first
        transcript_result = transcribe_url(request.url)

        # Segment content
        result = segment_content(
            transcript=transcript_result["transcript"],
            segments=transcript_result.get("segments"),
        )

        # Generate YouTube chapters
        yt_chapters = generate_youtube_chapters(result.chapters)

        return SegmentationResponse(
            total_topics=result.total_topics,
            total_chapters=result.total_chapters,
            avg_topic_duration=result.avg_topic_duration,
            topics=[
                TopicModel(
                    topic_id=t.topic_id,
                    title=t.title,
                    keywords=t.keywords[:5],
                    start_time=t.start_time,
                    end_time=t.end_time,
                    summary=t.summary,
                )
                for t in result.topics
            ],
            chapters=[
                ChapterModel(
                    chapter_id=ch.chapter_id,
                    title=ch.title,
                    start_time=ch.start_time,
                    end_time=ch.end_time,
                    duration=ch.duration,
                    summary=ch.summary,
                )
                for ch in result.chapters
            ],
            content_flow=result.content_flow,
            youtube_chapters=yt_chapters,
        )

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Segmentation failed: {e}")


@app.post("/v2/knowledge-graph")
async def build_video_knowledge_graph(request: ExtractRequest):
    """Build knowledge graph from a video's content."""
    try:
        logger.info(f"Building knowledge graph from: {request.url}")

        # Transcribe
        transcript_result = transcribe_url(request.url)

        # Extract knowledge
        knowledge = extract_knowledge(
            transcript=transcript_result["transcript"],
            title=transcript_result["title"],
        )

        # Extract entities and relationships
        entities = extract_entities(
            text=transcript_result["transcript"],
            video_id=transcript_result["video_id"],
            title=transcript_result["title"],
            knowledge=knowledge,
        )

        relationships = extract_relationships(
            entities=entities,
            text=transcript_result["transcript"],
            knowledge=knowledge,
            video_id=transcript_result["video_id"],
        )

        # Load existing graph
        config = get_config()
        graph_path = str(config.get_chroma_persist_path() / "knowledge_graph.json")
        existing_graph = load_knowledge_graph(graph_path)

        # Build/update graph
        graph = build_knowledge_graph(
            entities=entities,
            relationships=relationships,
            existing_graph=existing_graph,
        )
        save_knowledge_graph(graph, graph_path)

        stats = get_graph_statistics(graph)

        return KnowledgeGraphResponse(
            total_entities=stats["total_entities"],
            total_relationships=stats["total_relationships"],
            total_cross_references=stats["total_cross_references"],
            entity_types=stats["entity_types"],
            key_entities=[
                GraphEntityModel(
                    name=e["name"],
                    entity_type=e["type"],
                    frequency=e.get("connections", 1),
                )
                for e in stats["top_connected_entities"]
            ],
            videos_covered=stats["videos_covered"],
        )

    except Exception as e:
        logger.error(f"Knowledge graph building failed: {e}")
        raise HTTPException(status_code=400, detail=f"Knowledge graph building failed: {e}")


@app.get("/v2/knowledge-graph/stats")
async def get_knowledge_graph_statistics():
    """Get knowledge graph statistics."""
    try:
        config = get_config()
        graph_path = str(config.get_chroma_persist_path() / "knowledge_graph.json")
        graph = load_knowledge_graph(graph_path)

        if not graph:
            return {"message": "No knowledge graph found. Ingest videos first."}

        return get_graph_statistics(graph)

    except Exception as e:
        logger.error(f"Knowledge graph stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")


@app.get("/v2/knowledge-graph/query")
async def query_knowledge_graph(
    entity: str = None,
    entity_type: str = None,
    video_id: str = None,
    max_depth: int = 2,
):
    """Query the knowledge graph for entities and relationships."""
    try:
        config = get_config()
        graph_path = str(config.get_chroma_persist_path() / "knowledge_graph.json")
        graph = load_knowledge_graph(graph_path)

        if not graph:
            return {"message": "No knowledge graph found."}

        result = query_graph(
            graph=graph,
            entity_name=entity,
            entity_type=entity_type,
            video_id=video_id,
            max_depth=max_depth,
        )

        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                    "frequency": e.frequency,
                    "source_videos": e.source_videos,
                }
                for e in result["entities"][:50]
            ],
            "relationships": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type,
                    "weight": r.weight,
                }
                for r in result["relationships"][:50]
            ],
            "cross_references": [
                {
                    "source_video": cr.source_video,
                    "target_video": cr.target_video,
                    "shared_entities": cr.shared_entities,
                    "similarity": cr.similarity_score,
                }
                for cr in result["cross_references"][:20]
            ],
            "connected_entities": result["connected_entity_count"],
        }

    except Exception as e:
        logger.error(f"Knowledge graph query failed: {e}")
        raise HTTPException(status_code=400, detail=f"Query failed: {e}")


# ===== V3.0 Summarization Endpoints =====

@app.post("/v3/summarize", response_model=SummarizeResponse)
async def summarize_video_content(request: SummarizeRequest):
    """Advanced video summarization with chapter detection and key insights."""
    try:
        logger.info(f"Summarizing video: {request.url}")

        # Transcribe first
        transcript_result = transcribe_url(request.url)

        # Configure summarization
        chapter_cfg = ChapterDetectionConfig(
            max_chapters=request.max_chapters,
        )
        config = SummarizationConfig(
            chapter_config=chapter_cfg,
            max_insights=request.max_insights,
            min_insight_importance=request.min_insight_importance,
            enable_visual_integration=request.enable_visual,
            enable_speaker_integration=request.enable_speakers,
        )

        # Run summarization
        result = summarize_video(
            transcript=transcript_result["transcript"],
            video_id=transcript_result["video_id"],
            title=transcript_result["title"],
            segments=transcript_result.get("segments"),
            total_duration=transcript_result.get("duration"),
            config=config,
        )

        # Optional export
        export_content = None
        if request.export_format:
            if request.export_format == "markdown":
                export_content = export_summary_markdown(result)
            elif request.export_format == "json":
                export_content = export_summary_json(result)
            elif request.export_format in ("youtube", "srt", "vtt"):
                export_content = export_chapter_timestamps(result, format=request.export_format)

        # Save insights to database
        try:
            cfg = get_config()
            db_path = str(cfg.get_chroma_persist_path() / "insights_db")
            save_insights_to_db(result, db_path)
        except Exception as e:
            logger.warning(f"Failed to save insights to DB: {e}")

        summary = result.summary

        return SummarizeResponse(
            video_id=summary.video_id,
            title=summary.title,
            duration=summary.duration,
            executive_summary=summary.executive_summary,
            quality_score=summary.quality_score,
            overall_sentiment=summary.overall_sentiment.value,
            youtube_chapters=summary.youtube_chapters,
            content_flow=summary.content_flow,
            chapters=[
                EnhancedChapterModel(
                    chapter_id=ch.chapter_id,
                    title=ch.title,
                    start_time=ch.start_time,
                    end_time=ch.end_time,
                    duration=ch.duration,
                    summary=ch.summary,
                    sentiment=ch.sentiment.value,
                    sentiment_score=ch.sentiment_score,
                    has_visual_content=ch.has_visual_content,
                    scene_change_count=ch.scene_change_count,
                    confidence=ch.confidence,
                    key_insight_count=len(ch.key_insights),
                    slide_texts=ch.slide_texts,
                )
                for ch in summary.chapters
            ],
            key_insights=[
                InsightModel(
                    insight_id=ins.insight_id,
                    insight_type=ins.insight_type.value,
                    text=ins.text,
                    start_time=ins.start_time,
                    end_time=ins.end_time,
                    importance_score=ins.importance_score,
                    confidence=ins.confidence,
                    speaker_id=ins.speaker_id,
                    tags=ins.tags,
                )
                for ins in summary.key_insights
            ],
            action_items=[
                ActionItemModel(
                    action_id=item.action_id,
                    text=item.text,
                    priority=item.priority,
                    start_time=item.start_time,
                    confidence=item.confidence,
                )
                for item in summary.action_items
            ],
            top_quotes=[
                InsightModel(
                    insight_id=q.insight_id,
                    insight_type=q.insight_type.value,
                    text=q.text,
                    start_time=q.start_time,
                    end_time=q.end_time,
                    importance_score=q.importance_score,
                    confidence=q.confidence,
                    speaker_id=q.speaker_id,
                    tags=q.tags,
                )
                for q in summary.top_quotes
            ],
            sentiment_arc=[
                SentimentSegmentModel(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    label=seg.label.value,
                    score=seg.score,
                    keywords=seg.keywords,
                )
                for seg in summary.sentiment_arc
            ],
            quality_report=QualityReportModel(
                overall_quality_score=result.quality_report.get("overall_quality_score", 0),
                grade=result.quality_report.get("grade", "N/A"),
                chapter_coverage=result.quality_report.get("chapter_coverage", 0),
                insight_count=result.quality_report.get("insight_count", 0),
                insight_density_per_min=result.quality_report.get("insight_density_per_min", 0),
                insight_type_diversity=result.quality_report.get("insight_type_diversity", 0),
                sentiment_coverage=result.quality_report.get("sentiment_coverage", 0),
                action_item_count=result.quality_report.get("action_item_count", 0),
            ),
            processing_time_seconds=result.processing_time_seconds,
            export=export_content,
        )

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=400, detail=f"Summarization failed: {e}")


@app.post("/v3/insights/search", response_model=InsightSearchResponse)
async def search_video_insights(request: InsightSearchRequest):
    """Search the insights database across all processed videos."""
    try:
        cfg = get_config()
        db_path = str(cfg.get_chroma_persist_path() / "insights_db")

        results = search_insights(
            db_path=db_path,
            query=request.query,
            insight_type=request.insight_type,
            min_importance=request.min_importance,
            video_id=request.video_id,
            limit=request.limit,
        )

        return InsightSearchResponse(
            results=results,
            total_found=len(results),
        )

    except Exception as e:
        logger.error(f"Insight search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Insight search failed: {e}")


@app.post("/v3/chapters/export")
async def export_video_chapters(request: ExtractRequest, format: str = "youtube"):
    """Export chapter timestamps in various formats (youtube, srt, vtt, ffmpeg)."""
    try:
        logger.info(f"Exporting chapters for: {request.url} (format={format})")

        # Transcribe
        transcript_result = transcribe_url(request.url)

        # Summarize
        result = summarize_video(
            transcript=transcript_result["transcript"],
            video_id=transcript_result["video_id"],
            title=transcript_result["title"],
            segments=transcript_result.get("segments"),
        )

        # Export
        exported = export_chapter_timestamps(result, format=format)

        return {
            "video_id": transcript_result["video_id"],
            "title": transcript_result["title"],
            "format": format,
            "chapter_count": len(result.enhanced_chapters),
            "content": exported,
        }

    except Exception as e:
        logger.error(f"Chapter export failed: {e}")
        raise HTTPException(status_code=400, detail=f"Chapter export failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)