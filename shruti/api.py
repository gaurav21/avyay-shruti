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
from .models import (
    TranscribeRequest, TranscribeResponse,
    ExtractRequest, ExtractResponse,
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    BatchQueryRequest, BatchQueryResponse,
    DatabaseStatsResponse, HealthResponse,
    ErrorResponse, ConfigInfo,
    RelatedTopicsRequest, RelatedTopicsResponse,
    MetadataSearchRequest, MetadataSearchResponse
)
from .transcribe import transcribe_url
from .extract import extract_knowledge, extract_knowledge_batch
from .embeddings import add_video_to_vectorstore, get_database_stats
from .query import (
    query_knowledge, batch_query_knowledge, 
    get_related_topics, search_by_metadata
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        validate_config()
        setup_directories()
        logger.info("ŚRUTI API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "ŚRUTI (श्रुति) API",
        "version": __version__,
        "description": "Multilingual YouTube Knowledge Extractor",
        "docs": "/docs",
        "health": "/health"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)