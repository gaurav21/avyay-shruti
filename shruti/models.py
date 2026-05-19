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


# ===== V2.0 Multi-Modal Models =====

class VisualSlideText(BaseModel):
    """OCR-extracted text from a video frame."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    text: str = Field(..., description="Extracted text")
    confidence: float = Field(..., description="OCR confidence")
    language: Optional[str] = Field(None, description="Detected language")


class DiagramDetail(BaseModel):
    """Detected diagram information."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    diagram_type: str = Field(..., description="Diagram classification")
    description: str = Field(..., description="Description of the diagram")
    confidence: float = Field(0.0, description="Detection confidence")


class VisualAnalysisResponse(BaseModel):
    """Response model for visual analysis."""
    frames_extracted: int = Field(..., description="Number of frames analyzed")
    unique_slides: int = Field(..., description="Number of unique slides found")
    slide_texts: List[VisualSlideText] = Field(..., description="OCR-extracted texts")
    diagrams: List[DiagramDetail] = Field(..., description="Detected diagrams")
    scene_changes: int = Field(..., description="Number of scene changes")
    visual_summary: str = Field(..., description="Summary of visual analysis")


class SpeakerProfileModel(BaseModel):
    """Speaker profile information."""
    speaker_id: str = Field(..., description="Speaker identifier")
    speaker_label: Optional[str] = Field(None, description="Human-readable label")
    total_duration: float = Field(..., description="Total speaking time (seconds)")
    speaking_ratio: float = Field(..., description="Ratio of total duration")
    segment_count: int = Field(..., description="Number of speech segments")


class EmotionSummary(BaseModel):
    """Summary of emotion analysis."""
    dominant_emotion: str = Field(..., description="Most frequent emotion")
    emotion_distribution: Dict[str, int] = Field(..., description="Emotion counts")
    avg_sentiment_score: float = Field(..., description="Average sentiment score")
    overall_sentiment: str = Field(..., description="Overall sentiment: positive/negative/neutral")


class SpeakerAnalysisResponse(BaseModel):
    """Response model for speaker analysis."""
    total_speakers: int = Field(..., description="Number of detected speakers")
    dominant_speaker: Optional[str] = Field(None, description="Dominant speaker ID")
    speakers: List[SpeakerProfileModel] = Field(..., description="Speaker profiles")
    emotions: Optional[EmotionSummary] = Field(None, description="Emotion analysis summary")


class TopicModel(BaseModel):
    """A detected topic segment."""
    topic_id: int = Field(..., description="Topic identifier")
    title: str = Field(..., description="Topic title")
    keywords: List[str] = Field(..., description="Top keywords")
    start_time: float = Field(..., description="Start time (seconds)")
    end_time: float = Field(..., description="End time (seconds)")
    summary: str = Field("", description="Topic summary")


class ChapterModel(BaseModel):
    """An auto-detected chapter."""
    chapter_id: int = Field(..., description="Chapter identifier")
    title: str = Field(..., description="Chapter title")
    start_time: float = Field(..., description="Start time (seconds)")
    end_time: float = Field(..., description="End time (seconds)")
    duration: float = Field(..., description="Duration (seconds)")
    summary: str = Field("", description="Chapter summary")


class SegmentationResponse(BaseModel):
    """Response model for content segmentation."""
    total_topics: int = Field(..., description="Number of detected topics")
    total_chapters: int = Field(..., description="Number of detected chapters")
    avg_topic_duration: float = Field(..., description="Average topic duration")
    topics: List[TopicModel] = Field(..., description="Detected topics")
    chapters: List[ChapterModel] = Field(..., description="Detected chapters")
    content_flow: List[str] = Field(..., description="Ordered topic titles")
    youtube_chapters: Optional[str] = Field(None, description="YouTube-compatible chapter timestamps")


class GraphEntityModel(BaseModel):
    """Entity in the knowledge graph."""
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    frequency: int = Field(1, description="Occurrence frequency")
    description: str = Field("", description="Entity description")


class KnowledgeGraphResponse(BaseModel):
    """Response model for knowledge graph."""
    total_entities: int = Field(..., description="Total entities")
    total_relationships: int = Field(..., description="Total relationships")
    total_cross_references: int = Field(..., description="Total cross-references")
    entity_types: Dict[str, int] = Field(..., description="Entity type counts")
    key_entities: List[GraphEntityModel] = Field(..., description="Top entities")
    videos_covered: int = Field(0, description="Number of videos in graph")


class CrossReferenceModel(BaseModel):
    """Cross-reference between videos."""
    target_video: str = Field(..., description="Target video ID")
    shared_entities: List[str] = Field(..., description="Shared entity names")
    similarity: float = Field(..., description="Similarity score")
    ref_type: str = Field(..., description="Reference type")


class MultiModalRequest(BaseModel):
    """Request model for multi-modal analysis."""
    url: str = Field(..., description="YouTube video URL")
    enable_visual: bool = Field(True, description="Enable visual analysis")
    enable_speakers: bool = Field(True, description="Enable speaker identification")
    enable_segmentation: bool = Field(True, description="Enable topic segmentation")
    enable_knowledge_graph: bool = Field(True, description="Enable knowledge graph")
    enable_cross_references: bool = Field(True, description="Enable cross-references")
    frame_interval: float = Field(5.0, description="Frame capture interval (seconds)")
    max_frames: int = Field(200, description="Maximum frames to extract")


class MultiModalResponse(BaseModel):
    """Response model for multi-modal analysis."""
    video_id: str = Field(..., description="Video identifier")
    title: str = Field(..., description="Video title")
    language: str = Field(..., description="Primary language")
    knowledge: Dict[str, Any] = Field(..., description="V1 extracted knowledge")
    visual: Optional[VisualAnalysisResponse] = Field(None, description="Visual analysis")
    speakers: Optional[SpeakerAnalysisResponse] = Field(None, description="Speaker analysis")
    segmentation: Optional[SegmentationResponse] = Field(None, description="Content segmentation")
    knowledge_graph: Optional[KnowledgeGraphResponse] = Field(None, description="Knowledge graph")
    cross_references: List[CrossReferenceModel] = Field(default_factory=list, description="Cross-references")
    enriched_knowledge: Dict[str, Any] = Field(default_factory=dict, description="Merged enriched output")
    pipeline_stages: Dict[str, bool] = Field(..., description="Pipeline stage status")
    errors: List[str] = Field(default_factory=list, description="Pipeline errors")
    processing_time_seconds: float = Field(..., description="Total processing time")


# ===== V3.0 Summarization Models =====

class InsightModel(BaseModel):
    """A single extracted insight."""
    insight_id: str = Field(..., description="Insight identifier")
    insight_type: str = Field(..., description="Type: key_point, action_item, question, quote, definition, statistic, emphasis, transition, conclusion")
    text: str = Field(..., description="Insight text")
    start_time: float = Field(..., description="Start time (seconds)")
    end_time: float = Field(..., description="End time (seconds)")
    importance_score: float = Field(..., description="Importance score 0-1")
    confidence: float = Field(..., description="Detection confidence 0-1")
    speaker_id: Optional[str] = Field(None, description="Speaker ID if identified")
    tags: List[str] = Field(default_factory=list, description="Insight tags")


class ActionItemModel(BaseModel):
    """An identified action item."""
    action_id: str = Field(..., description="Action item identifier")
    text: str = Field(..., description="Action item text")
    priority: str = Field("medium", description="Priority: low, medium, high")
    start_time: float = Field(0.0, description="Start time (seconds)")
    confidence: float = Field(0.0, description="Detection confidence")


class SentimentSegmentModel(BaseModel):
    """Sentiment for a video segment."""
    start_time: float = Field(..., description="Start time (seconds)")
    end_time: float = Field(..., description="End time (seconds)")
    label: str = Field(..., description="Sentiment label")
    score: float = Field(..., description="Sentiment score -1 to 1")
    keywords: List[str] = Field(default_factory=list, description="Sentiment keywords")


class EnhancedChapterModel(BaseModel):
    """An enhanced chapter with multi-modal signals."""
    chapter_id: int = Field(..., description="Chapter identifier")
    title: str = Field(..., description="Chapter title")
    start_time: float = Field(..., description="Start time (seconds)")
    end_time: float = Field(..., description="End time (seconds)")
    duration: float = Field(..., description="Duration (seconds)")
    summary: str = Field("", description="Chapter summary")
    sentiment: str = Field("neutral", description="Chapter sentiment")
    sentiment_score: float = Field(0.0, description="Sentiment score")
    has_visual_content: bool = Field(False, description="Has slide/visual content")
    scene_change_count: int = Field(0, description="Scene changes in chapter")
    confidence: float = Field(0.0, description="Chapter boundary confidence")
    key_insight_count: int = Field(0, description="Number of key insights")
    slide_texts: List[str] = Field(default_factory=list, description="OCR slide texts")


class QualityReportModel(BaseModel):
    """Quality metrics for summarization."""
    overall_quality_score: float = Field(..., description="Overall quality 0-1")
    grade: str = Field(..., description="Letter grade A-F")
    chapter_coverage: float = Field(0.0, description="Chapter coverage ratio")
    insight_count: int = Field(0, description="Total insights")
    insight_density_per_min: float = Field(0.0, description="Insights per minute")
    insight_type_diversity: float = Field(0.0, description="Type diversity ratio")
    sentiment_coverage: float = Field(0.0, description="Sentiment coverage ratio")
    action_item_count: int = Field(0, description="Total action items")


class SummarizeRequest(BaseModel):
    """Request model for video summarization."""
    url: str = Field(..., description="YouTube video URL")
    max_insights: int = Field(50, description="Maximum insights to extract")
    min_insight_importance: float = Field(0.3, description="Minimum importance threshold")
    enable_visual: bool = Field(True, description="Enable visual analysis integration")
    enable_speakers: bool = Field(True, description="Enable speaker analysis integration")
    max_chapters: int = Field(15, description="Maximum chapters")
    export_format: Optional[str] = Field(None, description="Export format: markdown, json, youtube, srt, vtt")


class SummarizeResponse(BaseModel):
    """Response model for video summarization."""
    video_id: str = Field(..., description="Video identifier")
    title: str = Field(..., description="Video title")
    duration: float = Field(..., description="Video duration (seconds)")
    executive_summary: str = Field(..., description="Concise executive summary")
    quality_score: float = Field(..., description="Overall quality score 0-1")
    overall_sentiment: str = Field(..., description="Overall video sentiment")
    youtube_chapters: str = Field("", description="YouTube-compatible chapter timestamps")
    content_flow: List[str] = Field(default_factory=list, description="Content flow titles")
    chapters: List[EnhancedChapterModel] = Field(..., description="Enhanced chapters")
    key_insights: List[InsightModel] = Field(..., description="Key insights")
    action_items: List[ActionItemModel] = Field(..., description="Action items")
    top_quotes: List[InsightModel] = Field(..., description="Top quotable moments")
    sentiment_arc: List[SentimentSegmentModel] = Field(..., description="Sentiment arc")
    quality_report: QualityReportModel = Field(..., description="Quality metrics")
    processing_time_seconds: float = Field(..., description="Processing time")
    export: Optional[str] = Field(None, description="Exported content if format requested")


class InsightSearchRequest(BaseModel):
    """Request for searching insights database."""
    query: str = Field("", description="Text search query")
    insight_type: Optional[str] = Field(None, description="Filter by insight type")
    min_importance: float = Field(0.0, description="Minimum importance score")
    video_id: Optional[str] = Field(None, description="Filter by video ID")
    limit: int = Field(50, description="Maximum results")


class InsightSearchResponse(BaseModel):
    """Response for insights search."""
    results: List[Dict[str, Any]] = Field(..., description="Matching insights")
    total_found: int = Field(..., description="Total results found")