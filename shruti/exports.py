"""
Export & Integration module for ŚRUTI V3.0.
Provides chapter-based exports, searchable insights database,
and structured output formats for integration with external systems.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .insights import (
    ActionItem,
    Insight,
    InsightType,
    SentimentLabel,
    SentimentSegment,
)
from .summarize import EnhancedChapter, SummarizationResult, VideoSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def export_summary_markdown(
    result: SummarizationResult,
    include_timestamps: bool = True,
    include_sentiment: bool = True,
    include_action_items: bool = True,
    include_quotes: bool = True,
) -> str:
    """
    Export video summary as formatted Markdown.

    Args:
        result: SummarizationResult from summarize_video()
        include_timestamps: Include YouTube-compatible timestamps
        include_sentiment: Include sentiment analysis
        include_action_items: Include action items section
        include_quotes: Include notable quotes section

    Returns:
        Formatted Markdown string
    """
    summary = result.summary
    lines: List[str] = []

    # Header
    lines.append(f"# {summary.title}")
    lines.append("")
    lines.append(f"**Duration:** {_format_duration(summary.duration)}")
    lines.append(f"**Quality Score:** {summary.quality_score:.0%} ({result.quality_report.get('grade', 'N/A')})")
    lines.append(f"**Overall Sentiment:** {summary.overall_sentiment.value.replace('_', ' ').title()}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary.executive_summary)
    lines.append("")

    # Chapters
    lines.append("## Chapters")
    lines.append("")
    for ch in summary.chapters:
        timestamp = _format_timestamp(ch.start_time)
        lines.append(f"### {ch.chapter_id + 1}. {ch.title} [{timestamp}]")
        lines.append("")
        if ch.summary:
            lines.append(ch.summary[:300])
            lines.append("")
        if include_sentiment and ch.sentiment != SentimentLabel.NEUTRAL:
            lines.append(f"*Sentiment: {ch.sentiment.value.replace('_', ' ').title()} ({ch.sentiment_score:+.2f})*")
            lines.append("")
        if ch.key_insights:
            lines.append("**Key Points:**")
            for ins in ch.key_insights[:5]:
                lines.append(f"- {ins.text[:150]}")
            lines.append("")
        if ch.slide_texts:
            lines.append("**Slide Content:**")
            for slide in ch.slide_texts[:3]:
                lines.append(f"- {slide[:100]}")
            lines.append("")

    # YouTube Chapters
    if include_timestamps and summary.youtube_chapters:
        lines.append("## YouTube Chapters")
        lines.append("")
        lines.append("```")
        lines.append(summary.youtube_chapters)
        lines.append("```")
        lines.append("")

    # Key Insights
    if summary.key_insights:
        lines.append("## Key Insights")
        lines.append("")
        for ins in sorted(summary.key_insights, key=lambda i: i.importance_score, reverse=True)[:15]:
            badge = _insight_type_badge(ins.insight_type)
            timestamp = _format_timestamp(ins.start_time)
            lines.append(f"- {badge} [{timestamp}] {ins.text[:200]}")
        lines.append("")

    # Action Items
    if include_action_items and summary.action_items:
        lines.append("## Action Items")
        lines.append("")
        for item in summary.action_items:
            priority_badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.priority, "⚪")
            lines.append(f"- {priority_badge} **[{item.priority.upper()}]** {item.text[:200]}")
        lines.append("")

    # Notable Quotes
    if include_quotes and summary.top_quotes:
        lines.append("## Notable Quotes")
        lines.append("")
        for quote in summary.top_quotes[:10]:
            timestamp = _format_timestamp(quote.start_time)
            speaker = f" — {quote.speaker_id}" if quote.speaker_id else ""
            lines.append(f'> "{quote.text}"{speaker} [{timestamp}]')
            lines.append("")

    # Sentiment Arc
    if include_sentiment and summary.sentiment_arc:
        lines.append("## Sentiment Arc")
        lines.append("")
        for seg in summary.sentiment_arc:
            emoji = _sentiment_emoji(seg.label)
            timestamp = _format_timestamp(seg.start_time)
            lines.append(f"- {emoji} [{timestamp}] {seg.label.value.replace('_', ' ').title()} ({seg.score:+.2f})")
        lines.append("")

    # Quality Report
    lines.append("## Quality Report")
    lines.append("")
    qr = result.quality_report
    lines.append(f"- **Overall Score:** {qr.get('overall_quality_score', 0):.0%} (Grade: {qr.get('grade', 'N/A')})")
    lines.append(f"- **Chapter Coverage:** {qr.get('chapter_coverage', 0):.0%}")
    lines.append(f"- **Insight Density:** {qr.get('insight_density_per_min', 0):.1f} per minute")
    lines.append(f"- **Type Diversity:** {qr.get('insight_type_diversity', 0):.0%}")
    lines.append(f"- **Sentiment Coverage:** {qr.get('sentiment_coverage', 0):.0%}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_summary_json(
    result: SummarizationResult,
    pretty: bool = True,
) -> str:
    """
    Export video summary as JSON for API responses and storage.

    Args:
        result: SummarizationResult from summarize_video()
        pretty: Pretty-print JSON

    Returns:
        JSON string
    """
    summary = result.summary

    data = {
        "video_id": summary.video_id,
        "title": summary.title,
        "duration": summary.duration,
        "executive_summary": summary.executive_summary,
        "quality_score": summary.quality_score,
        "overall_sentiment": summary.overall_sentiment.value,
        "youtube_chapters": summary.youtube_chapters,
        "content_flow": summary.content_flow,
        "chapters": [
            {
                "chapter_id": ch.chapter_id,
                "title": ch.title,
                "start_time": ch.start_time,
                "end_time": ch.end_time,
                "duration": ch.duration,
                "summary": ch.summary,
                "sentiment": ch.sentiment.value,
                "sentiment_score": ch.sentiment_score,
                "has_visual_content": ch.has_visual_content,
                "scene_change_count": ch.scene_change_count,
                "confidence": ch.confidence,
                "key_insight_count": len(ch.key_insights),
                "slide_texts": ch.slide_texts,
            }
            for ch in summary.chapters
        ],
        "key_insights": [
            {
                "insight_id": ins.insight_id,
                "type": ins.insight_type.value,
                "text": ins.text,
                "start_time": ins.start_time,
                "end_time": ins.end_time,
                "importance_score": ins.importance_score,
                "confidence": ins.confidence,
                "speaker_id": ins.speaker_id,
                "tags": ins.tags,
            }
            for ins in summary.key_insights
        ],
        "action_items": [
            {
                "action_id": item.action_id,
                "text": item.text,
                "priority": item.priority,
                "start_time": item.start_time,
                "confidence": item.confidence,
            }
            for item in summary.action_items
        ],
        "top_quotes": [
            {
                "text": q.text,
                "start_time": q.start_time,
                "speaker_id": q.speaker_id,
                "importance_score": q.importance_score,
            }
            for q in summary.top_quotes
        ],
        "sentiment_arc": [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "label": seg.label.value,
                "score": seg.score,
                "keywords": seg.keywords,
            }
            for seg in summary.sentiment_arc
        ],
        "quality_report": result.quality_report,
        "processing_time_seconds": result.processing_time_seconds,
    }

    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Chapter-based video export
# ---------------------------------------------------------------------------

def export_chapter_timestamps(
    result: SummarizationResult,
    format: str = "youtube",
) -> str:
    """
    Export chapter timestamps in various formats.

    Formats:
    - youtube: YouTube description format (0:00 Title)
    - srt: SRT subtitle format for chapters
    - vtt: WebVTT format
    - ffmpeg: ffmpeg metadata format for chapter embedding

    Returns:
        Formatted timestamp string
    """
    chapters = result.enhanced_chapters

    if format == "youtube":
        return result.summary.youtube_chapters

    elif format == "srt":
        lines = []
        for idx, ch in enumerate(chapters, 1):
            lines.append(str(idx))
            start = _format_srt_timestamp(ch.start_time)
            end = _format_srt_timestamp(ch.end_time)
            lines.append(f"{start} --> {end}")
            lines.append(ch.title)
            lines.append("")
        return "\n".join(lines)

    elif format == "vtt":
        lines = ["WEBVTT", ""]
        for ch in chapters:
            start = _format_vtt_timestamp(ch.start_time)
            end = _format_vtt_timestamp(ch.end_time)
            lines.append(f"{start} --> {end}")
            lines.append(ch.title)
            lines.append("")
        return "\n".join(lines)

    elif format == "ffmpeg":
        lines = [";FFMETADATA1"]
        for ch in chapters:
            lines.append("")
            lines.append("[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={int(ch.start_time * 1000)}")
            lines.append(f"END={int(ch.end_time * 1000)}")
            lines.append(f"title={ch.title}")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unsupported format: {format}")


def split_video_by_chapters(
    video_path: str,
    result: SummarizationResult,
    output_dir: str,
    format: str = "mp4",
) -> List[Dict[str, Any]]:
    """
    Split a video file into chapter-based segments using ffmpeg.

    Args:
        video_path: Path to source video
        result: SummarizationResult with chapters
        output_dir: Directory for output files
        format: Output video format

    Returns:
        List of dicts with chapter info and output paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = []

    for ch in result.enhanced_chapters:
        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', ch.title)[:50].strip()
        safe_title = re.sub(r'\s+', '_', safe_title)
        output_path = os.path.join(
            output_dir,
            f"chapter_{ch.chapter_id:02d}_{safe_title}.{format}"
        )

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(ch.buffer_start),
            "-to", str(ch.buffer_end),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            output_path,
            "-y", "-loglevel", "warning",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
            outputs.append({
                "chapter_id": ch.chapter_id,
                "title": ch.title,
                "start_time": ch.start_time,
                "end_time": ch.end_time,
                "output_path": output_path,
                "success": True,
            })
            logger.info(f"Exported chapter {ch.chapter_id}: {ch.title}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Failed to export chapter {ch.chapter_id}: {e}")
            outputs.append({
                "chapter_id": ch.chapter_id,
                "title": ch.title,
                "output_path": output_path,
                "success": False,
                "error": str(e),
            })

    return outputs


# ---------------------------------------------------------------------------
# Insights database
# ---------------------------------------------------------------------------

def save_insights_to_db(
    result: SummarizationResult,
    db_path: str,
) -> Dict[str, Any]:
    """
    Save insights to a JSON-based searchable database.
    Each video's insights are stored as a separate entry,
    indexed by video_id for efficient retrieval.

    Args:
        result: SummarizationResult
        db_path: Path to the insights database directory

    Returns:
        Dict with save status
    """
    Path(db_path).mkdir(parents=True, exist_ok=True)

    video_id = result.summary.video_id
    db_file = os.path.join(db_path, "insights_index.json")

    # Load existing index
    index: Dict[str, Any] = {}
    if os.path.exists(db_file):
        try:
            with open(db_file, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, IOError):
            index = {}

    # Build entry
    entry = {
        "video_id": video_id,
        "title": result.summary.title,
        "duration": result.summary.duration,
        "quality_score": result.summary.quality_score,
        "overall_sentiment": result.summary.overall_sentiment.value,
        "chapter_count": len(result.enhanced_chapters),
        "insight_count": result.insights.total_insights,
        "action_item_count": len(result.insights.action_items),
        "insight_types": result.insights.insight_type_counts,
        "chapters": [
            {"title": ch.title, "start": ch.start_time, "end": ch.end_time}
            for ch in result.enhanced_chapters
        ],
        "insights": [
            {
                "id": ins.insight_id,
                "type": ins.insight_type.value,
                "text": ins.text[:300],
                "start_time": ins.start_time,
                "importance": ins.importance_score,
                "tags": ins.tags,
            }
            for ins in result.insights.insights
        ],
        "action_items": [
            {
                "id": item.action_id,
                "text": item.text[:300],
                "priority": item.priority,
            }
            for item in result.insights.action_items
        ],
    }

    # Update index
    if "videos" not in index:
        index["videos"] = {}
    index["videos"][video_id] = entry
    index["total_videos"] = len(index["videos"])
    index["total_insights"] = sum(
        v.get("insight_count", 0) for v in index["videos"].values()
    )

    # Save
    with open(db_file, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    # Also save individual video file
    video_file = os.path.join(db_path, f"{video_id}.json")
    with open(video_file, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved insights for {video_id} to {db_path}")

    return {
        "video_id": video_id,
        "insights_saved": entry["insight_count"],
        "db_path": db_path,
        "total_videos_in_db": index["total_videos"],
    }


def search_insights(
    db_path: str,
    query: str = "",
    insight_type: Optional[str] = None,
    min_importance: float = 0.0,
    video_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Search the insights database.

    Args:
        db_path: Path to insights database directory
        query: Text search query
        insight_type: Filter by insight type
        min_importance: Minimum importance score
        video_id: Filter by video ID
        limit: Maximum results

    Returns:
        List of matching insights
    """
    db_file = os.path.join(db_path, "insights_index.json")
    if not os.path.exists(db_file):
        return []

    try:
        with open(db_file, "r") as f:
            index = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    results: List[Dict[str, Any]] = []
    query_lower = query.lower()

    for vid, video_data in index.get("videos", {}).items():
        if video_id and vid != video_id:
            continue

        for ins in video_data.get("insights", []):
            # Filter by type
            if insight_type and ins.get("type") != insight_type:
                continue

            # Filter by importance
            if ins.get("importance", 0) < min_importance:
                continue

            # Text search
            if query_lower and query_lower not in ins.get("text", "").lower():
                # Also search tags
                if not any(query_lower in tag.lower() for tag in ins.get("tags", [])):
                    continue

            results.append({
                **ins,
                "video_id": vid,
                "video_title": video_data.get("title", ""),
            })

            if len(results) >= limit:
                return results

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_srt_timestamp(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """Format timestamp for WebVTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _insight_type_badge(insight_type: InsightType) -> str:
    """Get an emoji badge for insight type."""
    badges = {
        InsightType.KEY_POINT: "💡",
        InsightType.ACTION_ITEM: "✅",
        InsightType.QUESTION: "❓",
        InsightType.QUOTE: "💬",
        InsightType.DEFINITION: "📖",
        InsightType.STATISTIC: "📊",
        InsightType.EMPHASIS: "⚡",
        InsightType.TRANSITION: "➡️",
        InsightType.CONCLUSION: "🏁",
    }
    return badges.get(insight_type, "•")


def _sentiment_emoji(label: SentimentLabel) -> str:
    """Get emoji for sentiment label."""
    emojis = {
        SentimentLabel.VERY_POSITIVE: "😀",
        SentimentLabel.POSITIVE: "🙂",
        SentimentLabel.NEUTRAL: "😐",
        SentimentLabel.NEGATIVE: "😟",
        SentimentLabel.VERY_NEGATIVE: "😞",
    }
    return emojis.get(label, "😐")
