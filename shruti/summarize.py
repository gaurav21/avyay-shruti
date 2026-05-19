"""
Advanced Video Summarization Engine for ŚRUTI V3.0.
Combines chapter detection, key insights, multi-modal analysis,
and quality scoring into a comprehensive summarization pipeline.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .insights import (
    ActionItem,
    HighlightReel,
    Insight,
    InsightType,
    InsightsResult,
    SentimentLabel,
    SentimentSegment,
    extract_insights,
)
from .segments import (
    Chapter,
    SegmentationResult,
    TopicSegment,
    detect_chapters,
    generate_youtube_chapters,
    segment_content,
    segment_topics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnhancedChapter:
    """A chapter enhanced with multi-modal signals."""
    chapter_id: int
    title: str
    start_time: float
    end_time: float
    duration: float
    summary: str
    topic_ids: List[int] = field(default_factory=list)
    key_insights: List[Insight] = field(default_factory=list)
    sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    sentiment_score: float = 0.0
    has_visual_content: bool = False
    has_speaker_change: bool = False
    scene_change_count: int = 0
    slide_texts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    buffer_start: float = 0.0  # Buffer zone before chapter
    buffer_end: float = 0.0    # Buffer zone after chapter


@dataclass
class VideoSummary:
    """Complete video summary combining all analysis."""
    video_id: str
    title: str
    duration: float
    executive_summary: str
    chapters: List[EnhancedChapter]
    key_insights: List[Insight]
    action_items: List[ActionItem]
    top_quotes: List[Insight]
    sentiment_arc: List[SentimentSegment]
    overall_sentiment: SentimentLabel
    content_flow: List[str]
    youtube_chapters: str
    quality_score: float
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChapterDetectionConfig:
    """Configuration for chapter detection."""
    min_chapter_duration: float = 60.0
    max_chapters: int = 15
    buffer_seconds: float = 2.0
    use_scene_changes: bool = True
    use_speaker_changes: bool = True
    use_embeddings: bool = True
    similarity_threshold: float = 0.5
    min_topic_sentences: int = 3
    max_topics: int = 20


@dataclass
class SummarizationConfig:
    """Configuration for the full summarization pipeline."""
    chapter_config: ChapterDetectionConfig = field(default_factory=ChapterDetectionConfig)
    max_insights: int = 50
    min_insight_importance: float = 0.3
    max_summary_length: int = 500
    enable_visual_integration: bool = True
    enable_speaker_integration: bool = True
    enable_sentiment: bool = True


@dataclass
class SummarizationResult:
    """Complete result from the summarization pipeline."""
    summary: VideoSummary
    segmentation: SegmentationResult
    insights: InsightsResult
    raw_chapters: List[Chapter]
    enhanced_chapters: List[EnhancedChapter]
    quality_report: Dict[str, Any]
    processing_time_seconds: float


# ---------------------------------------------------------------------------
# Enhanced chapter detection
# ---------------------------------------------------------------------------

def detect_chapters_enhanced(
    transcript: str,
    segments: Optional[List[Dict]] = None,
    scene_changes: Optional[List[float]] = None,
    speaker_changes: Optional[List[float]] = None,
    config: Optional[ChapterDetectionConfig] = None,
) -> Tuple[List[EnhancedChapter], SegmentationResult]:
    """
    Enhanced chapter detection combining multiple signals:
    - Transcript topic changes (embedding-based)
    - Scene change timestamps from visual analysis
    - Speaker changes from diarization
    - Smart buffer zones between chapters

    Args:
        transcript: Full transcript text
        segments: Optional timed transcript segments
        scene_changes: Timestamps of visual scene changes
        speaker_changes: Timestamps of speaker changes
        config: Chapter detection configuration

    Returns:
        Tuple of (enhanced chapters, segmentation result)
    """
    if config is None:
        config = ChapterDetectionConfig()

    logger.info("Running enhanced chapter detection")

    # Step 1: Topic segmentation
    seg_result = segment_content(
        transcript=transcript,
        segments=segments,
        min_topic_sentences=config.min_topic_sentences,
        max_topics=config.max_topics,
        min_chapter_duration=config.min_chapter_duration,
        max_chapters=config.max_chapters,
    )

    # Step 2: Refine chapter boundaries using additional signals
    enhanced_chapters = _refine_chapters_with_signals(
        chapters=seg_result.chapters,
        topics=seg_result.topics,
        scene_changes=scene_changes or [],
        speaker_changes=speaker_changes or [],
        buffer_seconds=config.buffer_seconds,
    )

    logger.info(f"Enhanced chapter detection: {len(enhanced_chapters)} chapters")
    return enhanced_chapters, seg_result


def _refine_chapters_with_signals(
    chapters: List[Chapter],
    topics: List[TopicSegment],
    scene_changes: List[float],
    speaker_changes: List[float],
    buffer_seconds: float = 2.0,
) -> List[EnhancedChapter]:
    """
    Refine chapter boundaries using multi-modal signals.
    Adjusts chapter start/end times to align with natural breaks.
    """
    enhanced: List[EnhancedChapter] = []

    for ch in chapters:
        # Find nearest scene change to chapter boundary
        adjusted_start = ch.start_time
        adjusted_end = ch.end_time

        if scene_changes:
            nearest_start = _find_nearest_timestamp(ch.start_time, scene_changes, window=10.0)
            if nearest_start is not None:
                adjusted_start = nearest_start

            nearest_end = _find_nearest_timestamp(ch.end_time, scene_changes, window=10.0)
            if nearest_end is not None and nearest_end > adjusted_start:
                adjusted_end = nearest_end

        # Count scene changes within chapter
        sc_count = sum(1 for sc in scene_changes if adjusted_start <= sc <= adjusted_end)

        # Check for speaker changes
        has_speaker_change = any(
            adjusted_start <= sc <= adjusted_end for sc in speaker_changes
        )

        # Compute buffer zones
        buffer_start = max(0, adjusted_start - buffer_seconds)
        buffer_end = adjusted_end + buffer_seconds

        # Get associated topics
        topic_ids = ch.topic_ids if ch.topic_ids else [t.topic_id for t in topics if t.start_time >= adjusted_start and t.end_time <= adjusted_end]

        enhanced.append(EnhancedChapter(
            chapter_id=ch.chapter_id,
            title=ch.title,
            start_time=adjusted_start,
            end_time=adjusted_end,
            duration=adjusted_end - adjusted_start,
            summary=ch.summary,
            topic_ids=topic_ids,
            has_speaker_change=has_speaker_change,
            scene_change_count=sc_count,
            confidence=_chapter_confidence(ch, sc_count, has_speaker_change),
            buffer_start=buffer_start,
            buffer_end=buffer_end,
        ))

    return enhanced


def _find_nearest_timestamp(
    target: float,
    timestamps: List[float],
    window: float = 10.0,
) -> Optional[float]:
    """Find the nearest timestamp within a window."""
    candidates = [t for t in timestamps if abs(t - target) <= window]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - target))


def _chapter_confidence(
    chapter: Chapter,
    scene_changes: int,
    has_speaker_change: bool,
) -> float:
    """Calculate confidence score for a chapter boundary."""
    confidence = 0.5  # Base confidence from topic segmentation

    # Scene change at boundary boosts confidence
    if scene_changes > 0:
        confidence += 0.2

    # Speaker change at boundary boosts confidence
    if has_speaker_change:
        confidence += 0.15

    # Longer chapters are generally more confident
    if chapter.duration >= 60:
        confidence += 0.1
    elif chapter.duration >= 30:
        confidence += 0.05

    return min(1.0, confidence)


# ---------------------------------------------------------------------------
# Integrate insights into chapters
# ---------------------------------------------------------------------------

def _enrich_chapters_with_insights(
    chapters: List[EnhancedChapter],
    insights: InsightsResult,
) -> List[EnhancedChapter]:
    """Assign insights and sentiment to their respective chapters."""
    for ch in chapters:
        # Assign insights to chapter by timestamp
        ch.key_insights = [
            ins for ins in insights.insights
            if ch.start_time <= ins.start_time <= ch.end_time
        ]

        # Compute chapter sentiment
        chapter_sentiments = [
            seg for seg in insights.sentiment_segments
            if ch.start_time <= seg.start_time <= ch.end_time
        ]
        if chapter_sentiments:
            avg_score = sum(s.score for s in chapter_sentiments) / len(chapter_sentiments)
            ch.sentiment_score = avg_score
            if avg_score > 0.15:
                ch.sentiment = SentimentLabel.POSITIVE
            elif avg_score < -0.15:
                ch.sentiment = SentimentLabel.NEGATIVE
            else:
                ch.sentiment = SentimentLabel.NEUTRAL

    return chapters


def _enrich_chapters_with_visuals(
    chapters: List[EnhancedChapter],
    slide_texts: Optional[List[Dict]] = None,
) -> List[EnhancedChapter]:
    """Enrich chapters with visual analysis data."""
    if not slide_texts:
        return chapters

    for ch in chapters:
        matching_slides = [
            s for s in slide_texts
            if ch.start_time <= s.get("timestamp", 0) <= ch.end_time
        ]
        if matching_slides:
            ch.has_visual_content = True
            ch.slide_texts = [s.get("text", "")[:200] for s in matching_slides[:5]]

    return chapters


# ---------------------------------------------------------------------------
# Executive summary generation
# ---------------------------------------------------------------------------

def generate_executive_summary(
    title: str,
    chapters: List[EnhancedChapter],
    insights: InsightsResult,
    max_length: int = 500,
) -> str:
    """
    Generate a concise executive summary of the video.
    Combines chapter summaries with key insights.
    """
    parts = []

    # Title context
    parts.append(f'"{title}" covers {len(chapters)} main sections.')

    # Chapter overview
    chapter_summaries = []
    for ch in chapters[:5]:
        if ch.summary:
            chapter_summaries.append(f"• {ch.title}: {ch.summary[:100]}")
    if chapter_summaries:
        parts.append("Key sections:\n" + "\n".join(chapter_summaries))

    # Top insights
    top_insights = sorted(
        insights.insights, key=lambda i: i.importance_score, reverse=True
    )[:3]
    if top_insights:
        insight_texts = [f"• {ins.text[:100]}" for ins in top_insights]
        parts.append("Key takeaways:\n" + "\n".join(insight_texts))

    # Action items
    if insights.action_items:
        action_texts = [f"• {a.text[:80]}" for a in insights.action_items[:3]]
        parts.append("Action items:\n" + "\n".join(action_texts))

    # Overall sentiment
    parts.append(f"Overall tone: {insights.overall_sentiment.value.replace('_', ' ')}")

    summary = "\n\n".join(parts)
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."

    return summary


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_summarization_quality(
    chapters: List[EnhancedChapter],
    insights: InsightsResult,
    transcript_length: int,
    total_duration: float,
) -> Dict[str, Any]:
    """
    Compute quality metrics for the summarization output.

    Evaluates:
    - Chapter coverage (% of video covered)
    - Insight density (insights per minute)
    - Diversity of insight types
    - Sentiment arc completeness
    - Overall quality score
    """
    report: Dict[str, Any] = {}

    # Chapter coverage
    if chapters and total_duration > 0:
        covered_duration = sum(ch.duration for ch in chapters)
        report["chapter_coverage"] = min(1.0, covered_duration / total_duration)
        report["chapter_count"] = len(chapters)
        report["avg_chapter_duration"] = covered_duration / len(chapters)
    else:
        report["chapter_coverage"] = 0.0
        report["chapter_count"] = 0
        report["avg_chapter_duration"] = 0.0

    # Insight quality
    if insights.insights:
        report["insight_count"] = len(insights.insights)
        report["insight_density_per_min"] = len(insights.insights) / max(total_duration / 60, 1)
        report["avg_insight_importance"] = (
            sum(i.importance_score for i in insights.insights) / len(insights.insights)
        )
        report["avg_insight_confidence"] = (
            sum(i.confidence for i in insights.insights) / len(insights.insights)
        )
    else:
        report["insight_count"] = 0
        report["insight_density_per_min"] = 0.0
        report["avg_insight_importance"] = 0.0
        report["avg_insight_confidence"] = 0.0

    # Type diversity
    types_found = len(set(i.insight_type for i in insights.insights)) if insights.insights else 0
    report["insight_type_diversity"] = types_found / len(InsightType)

    # Sentiment coverage
    if insights.sentiment_segments and total_duration > 0:
        sentiment_coverage = sum(
            s.end_time - s.start_time for s in insights.sentiment_segments
        ) / total_duration
        report["sentiment_coverage"] = min(1.0, sentiment_coverage)
    else:
        report["sentiment_coverage"] = 0.0

    # Action items
    report["action_item_count"] = len(insights.action_items)

    # Overall quality score (weighted average)
    weights = {
        "chapter_coverage": 0.25,
        "insight_type_diversity": 0.2,
        "avg_insight_confidence": 0.2,
        "sentiment_coverage": 0.15,
        "avg_insight_importance": 0.2,
    }
    overall = sum(
        report.get(k, 0) * w for k, w in weights.items()
    )
    report["overall_quality_score"] = min(1.0, overall)

    # Grade
    if overall >= 0.8:
        report["grade"] = "A"
    elif overall >= 0.6:
        report["grade"] = "B"
    elif overall >= 0.4:
        report["grade"] = "C"
    elif overall >= 0.2:
        report["grade"] = "D"
    else:
        report["grade"] = "F"

    return report


# ---------------------------------------------------------------------------
# Full summarization pipeline
# ---------------------------------------------------------------------------

def summarize_video(
    transcript: str,
    video_id: str = "",
    title: str = "",
    segments: Optional[List[Dict]] = None,
    scene_changes: Optional[List[float]] = None,
    speaker_changes: Optional[List[float]] = None,
    speaker_segments: Optional[List[Dict]] = None,
    slide_texts: Optional[List[Dict]] = None,
    total_duration: Optional[float] = None,
    config: Optional[SummarizationConfig] = None,
) -> SummarizationResult:
    """
    Complete video summarization pipeline.

    Combines:
    1. Enhanced chapter detection with multi-modal signals
    2. Key insights extraction (emphasis, action items, quotes, etc.)
    3. Sentiment analysis across the video
    4. Quality scoring and reporting

    Args:
        transcript: Full transcript text
        video_id: Video identifier
        title: Video title
        segments: Timed transcript segments [{start, end, text}, ...]
        scene_changes: Scene change timestamps from visual analysis
        speaker_changes: Speaker change timestamps from diarization
        speaker_segments: Speaker-attributed segments [{speaker_id, start, end, text}]
        slide_texts: OCR slide texts [{timestamp, text, confidence}]
        total_duration: Total video duration in seconds
        config: Summarization configuration

    Returns:
        SummarizationResult with complete analysis
    """
    start_time = time.time()

    if config is None:
        config = SummarizationConfig()

    if not video_id:
        video_id = hashlib.md5(transcript[:200].encode()).hexdigest()[:12]

    logger.info(f"Starting video summarization for '{title}' ({len(transcript)} chars)")

    # Estimate duration if not provided
    if total_duration is None:
        if segments:
            total_duration = max(s.get("end", 0) for s in segments) if segments else 0
        else:
            total_duration = len(transcript.split()) * 0.4  # ~0.4s per word

    # Step 1: Enhanced chapter detection
    enhanced_chapters, seg_result = detect_chapters_enhanced(
        transcript=transcript,
        segments=segments,
        scene_changes=scene_changes if config.enable_visual_integration else None,
        speaker_changes=speaker_changes if config.enable_speaker_integration else None,
        config=config.chapter_config,
    )

    # Step 2: Extract insights
    insights = extract_insights(
        transcript=transcript,
        segments=segments,
        video_id=video_id,
        title=title,
        speaker_segments=speaker_segments,
        max_insights=config.max_insights,
        min_importance=config.min_insight_importance,
    )

    # Step 3: Enrich chapters with insights and visuals
    enhanced_chapters = _enrich_chapters_with_insights(enhanced_chapters, insights)
    if config.enable_visual_integration:
        enhanced_chapters = _enrich_chapters_with_visuals(enhanced_chapters, slide_texts)

    # Step 4: Generate executive summary
    executive_summary = generate_executive_summary(
        title=title,
        chapters=enhanced_chapters,
        insights=insights,
        max_length=config.max_summary_length,
    )

    # Step 5: Generate YouTube chapters
    yt_chapters = generate_youtube_chapters(seg_result.chapters)

    # Step 6: Quality scoring
    quality_report = compute_summarization_quality(
        chapters=enhanced_chapters,
        insights=insights,
        transcript_length=len(transcript),
        total_duration=total_duration,
    )

    # Build final summary
    video_summary = VideoSummary(
        video_id=video_id,
        title=title,
        duration=total_duration,
        executive_summary=executive_summary,
        chapters=enhanced_chapters,
        key_insights=insights.insights,
        action_items=insights.action_items,
        top_quotes=insights.highlight_reel.top_quotes,
        sentiment_arc=insights.sentiment_segments,
        overall_sentiment=insights.overall_sentiment,
        content_flow=seg_result.content_flow,
        youtube_chapters=yt_chapters,
        quality_score=quality_report["overall_quality_score"],
        processing_metadata={
            "transcript_length": len(transcript),
            "total_duration": total_duration,
            "config": {
                "max_insights": config.max_insights,
                "min_insight_importance": config.min_insight_importance,
                "enable_visual": config.enable_visual_integration,
                "enable_speakers": config.enable_speaker_integration,
            },
        },
    )

    processing_time = time.time() - start_time
    logger.info(
        f"Summarization complete in {processing_time:.1f}s: "
        f"{len(enhanced_chapters)} chapters, {insights.total_insights} insights, "
        f"quality={quality_report['overall_quality_score']:.2f} ({quality_report['grade']})"
    )

    return SummarizationResult(
        summary=video_summary,
        segmentation=seg_result,
        insights=insights,
        raw_chapters=seg_result.chapters,
        enhanced_chapters=enhanced_chapters,
        quality_report=quality_report,
        processing_time_seconds=processing_time,
    )
