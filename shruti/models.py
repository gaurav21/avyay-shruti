"""
Pydantic models for ŚRUTI API.
Comprehensive models for all endpoints including health, transcription, extraction, and RAG.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Package version") 
    config_valid: bool = Field(..., description="Whether configuration is valid")
    database_accessible: bool = Field(..., description="Whether vector database is accessible")


class ConfigInfo(BaseModel):
    """Configuration information."""
    embedding_model: str = Field(..., description="Embedding model name")
    llm_model: str = Field(..., description="Language model name")
    chunk_size: int = Field(..., description="Text chunk size")
    chunk_overlap: int = Field(..., description="Text chunk overlap")
    chroma_persist_dir: str = Field(..., description="Chroma persistence directory")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    details: str = Field(..., description="Error details")
    error_type: str = Field(..., description="Error type")


# Transcription models
class TranscribeRequest(BaseModel):
    """Request model for video transcription."""
    url: str = Field(..., description="YouTube video URL")


class Segment(BaseModel):
    """Transcript segment with timing."""
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Segment text")


class TranscribeResponse(BaseModel):
    """Response model for video transcription."""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    transcript: str = Field(..., description="Full transcript text")
    language: str = Field(..., description="Detected language")
    source: str = Field(..., description="Transcript source (whisper/captions)")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    url: str = Field(..., description="Original video URL")
    segments: Optional[List[Segment]] = Field(None, description="Transcript segments with timing")


# Knowledge extraction models  
class ExtractRequest(BaseModel):
    """Request model for knowledge extraction."""
    url: str = Field(..., description="YouTube video URL")


class ExtractResponse(BaseModel):
    """Response model for knowledge extraction."""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    knowledge: Dict[str, Any] = Field(..., description="Extracted structured knowledge")
    transcript_source: str = Field(..., description="Transcript source")
    language: str = Field(..., description="Content language")


# Ingestion models
class IngestRequest(BaseModel):
    """Request model for video ingestion."""
    url: str = Field(..., description="YouTube video URL")
    persist_dir: Optional[str] = Field(None, description="Custom persistence directory")


class IngestResponse(BaseModel):
    """Response model for video ingestion."""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    chunks_stored: int = Field(..., description="Number of text chunks stored")
    language: str = Field(..., description="Content language")
    transcript_source: str = Field(..., description="Transcript source")
    persist_dir: str = Field(..., description="Persistence directory used")


# Query models
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="Question to ask")
    top_k: int = Field(5, description="Number of relevant chunks to retrieve")
    persist_dir: Optional[str] = Field(None, description="Custom persistence directory")
    include_metadata: bool = Field(True, description="Include chunk metadata in response")
    temperature: float = Field(0.1, description="LLM temperature for answer generation")


class Source(BaseModel):
    """Source information for RAG response."""
    video_id: str = Field(..., description="Source video ID")
    title: str = Field(..., description="Source video title")
    chunk_text: str = Field(..., description="Relevant text chunk")
    similarity_score: float = Field(..., description="Similarity to query")
    timestamp: Optional[float] = Field(None, description="Approximate timestamp in video")
    chunk_index: int = Field(..., description="Chunk index in video")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source chunks used")
    confidence: str = Field(..., description="Answer confidence level")
    chunks_found: int = Field(..., description="Number of relevant chunks found")
    avg_similarity: float = Field(..., description="Average similarity score")
    query: str = Field(..., description="Original query")


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    queries: List[str] = Field(..., description="List of questions")
    top_k: int = Field(5, description="Number of chunks per query")
    persist_dir: Optional[str] = Field(None, description="Custom persistence directory")


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""
    results: List[QueryResponse] = Field(..., description="Individual query results")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")


# Database and metadata models
class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics."""
    total_chunks: int = Field(..., description="Total number of stored chunks")
    unique_videos: int = Field(..., description="Number of unique videos")
    languages: List[str] = Field(..., description="Languages present in database")
    persist_dir: str = Field(..., description="Database persistence directory")


class RelatedTopicsRequest(BaseModel):
    """Request model for finding related topics."""
    topic: str = Field(..., description="Topic to find relations for")
    top_k: int = Field(10, description="Number of related topics to return")
    persist_dir: Optional[str] = Field(None, description="Custom persistence directory")


class RelatedTopic(BaseModel):
    """Related topic information."""
    topic: str = Field(..., description="Related topic")
    relevance_score: float = Field(..., description="Relevance score")
    video_count: int = Field(..., description="Number of videos containing this topic")


class RelatedTopicsResponse(BaseModel):
    """Response model for related topics."""
    topic: str = Field(..., description="Original topic")
    related_topics: List[RelatedTopic] = Field(..., description="Related topics found")


class MetadataSearchRequest(BaseModel):
    """Request model for metadata-based search."""
    filters: Dict[str, Any] = Field(..., description="Metadata filters to apply")
    limit: int = Field(50, description="Maximum number of results")
    persist_dir: Optional[str] = Field(None, description="Custom persistence directory")


class MetadataSearchResult(BaseModel):
    """Single metadata search result."""
    video_id: str = Field(..., description="Video ID")
    title: str = Field(..., description="Video title")
    chunk_text: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk index")
    metadata: Dict[str, Any] = Field(..., description="Full chunk metadata")


class MetadataSearchResponse(BaseModel):
    """Response model for metadata search."""
    results: List[MetadataSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")