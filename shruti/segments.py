"""
Topic segmentation and automated chapter detection module for ŚRUTI V2.0.
Segments transcripts into meaningful topics and auto-generates chapter markers.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TopicSegment:
    """A segment of content focused on a specific topic."""
    topic_id: int
    title: str
    summary: str
    start_time: float
    end_time: float
    start_index: int  # chunk/sentence index
    end_index: int
    text: str
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Chapter:
    """An auto-detected chapter/section in the video."""
    chapter_id: int
    title: str
    start_time: float
    end_time: float
    duration: float = 0.0
    summary: str = ""
    topic_ids: List[int] = field(default_factory=list)


@dataclass
class SegmentationResult:
    """Complete segmentation result."""
    topics: List[TopicSegment]
    chapters: List[Chapter]
    total_topics: int
    total_chapters: int
    avg_topic_duration: float
    content_flow: List[str]  # ordered list of topic titles showing flow


# ---------------------------------------------------------------------------
# Topic segmentation
# ---------------------------------------------------------------------------

def segment_topics(
    transcript: str,
    segments: Optional[List[Dict]] = None,
    min_topic_sentences: int = 3,
    max_topics: int = 20,
    similarity_threshold: float = 0.5,
    use_embeddings: bool = True,
) -> List[TopicSegment]:
    """
    Segment transcript into distinct topics using text similarity analysis.

    Strategy:
    1. Split into sentences
    2. Compute sentence embeddings
    3. Detect topic boundaries via cosine similarity drops
    4. Group sentences into topic segments
    5. Generate titles and keywords for each topic

    Args:
        transcript: Full transcript text
        segments: Optional timed segments for timestamp alignment
        min_topic_sentences: Minimum sentences per topic
        max_topics: Maximum number of topics to detect
        similarity_threshold: Threshold for topic boundary detection
        use_embeddings: Use sentence embeddings (True) or keyword-based (False)

    Returns:
        List of TopicSegment objects
    """
    # Split into sentences
    sentences = _split_into_sentences(transcript)
    if len(sentences) < min_topic_sentences:
        return [_single_topic(transcript, sentences, segments)]

    # Detect topic boundaries
    if use_embeddings:
        try:
            boundaries = _detect_boundaries_embeddings(
                sentences, similarity_threshold, min_topic_sentences
            )
        except Exception as e:
            logger.warning(f"Embedding-based segmentation failed: {e}, falling back to keyword-based")
            boundaries = _detect_boundaries_keyword(sentences, min_topic_sentences)
    else:
        boundaries = _detect_boundaries_keyword(sentences, min_topic_sentences)

    # Limit number of topics
    if len(boundaries) > max_topics:
        # Keep only the strongest boundaries
        boundaries = boundaries[:max_topics]

    # Build topic segments
    topics = _build_topic_segments(sentences, boundaries, segments)

    logger.info(f"Detected {len(topics)} topics from {len(sentences)} sentences")
    return topics


def _detect_boundaries_embeddings(
    sentences: List[str],
    threshold: float = 0.5,
    min_segment_size: int = 3,
) -> List[int]:
    """
    Detect topic boundaries using sentence embedding similarity.
    Uses a sliding window approach with cosine similarity.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load multilingual model (same as ŚRUTI embeddings)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Encode sentences
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Compute cosine similarity between consecutive sentence windows
    window_size = min(3, len(sentences) // 4)
    if window_size < 1:
        window_size = 1

    similarities = []
    for i in range(window_size, len(embeddings) - window_size):
        # Average embedding of preceding window
        prev_window = np.mean(embeddings[max(0, i - window_size):i], axis=0)
        # Average embedding of following window
        next_window = np.mean(embeddings[i:min(len(embeddings), i + window_size)], axis=0)

        # Cosine similarity
        sim = np.dot(prev_window, next_window) / (
            np.linalg.norm(prev_window) * np.linalg.norm(next_window) + 1e-8
        )
        similarities.append((i, float(sim)))

    # Find boundaries where similarity drops below threshold
    boundaries = [0]  # Always start with index 0

    for idx, sim in similarities:
        if sim < threshold:
            # Check minimum segment size
            if idx - boundaries[-1] >= min_segment_size:
                boundaries.append(idx)

    return boundaries


def _detect_boundaries_keyword(
    sentences: List[str],
    min_segment_size: int = 3,
) -> List[int]:
    """
    Detect topic boundaries using keyword overlap (no ML dependencies).
    """
    boundaries = [0]

    for i in range(min_segment_size, len(sentences)):
        # Compare keyword sets of preceding and following windows
        window = min(3, i)
        prev_keywords = _extract_keywords(" ".join(sentences[max(0, i - window):i]))
        next_keywords = _extract_keywords(" ".join(sentences[i:min(len(sentences), i + window)]))

        if prev_keywords and next_keywords:
            overlap = len(prev_keywords & next_keywords)
            union = len(prev_keywords | next_keywords)
            similarity = overlap / max(union, 1)

            if similarity < 0.2 and (i - boundaries[-1]) >= min_segment_size:
                boundaries.append(i)

    return boundaries


def _build_topic_segments(
    sentences: List[str],
    boundaries: List[int],
    timed_segments: Optional[List[Dict]],
) -> List[TopicSegment]:
    """Build TopicSegment objects from sentence boundaries."""
    topics: List[TopicSegment] = []

    # Add end boundary
    all_boundaries = sorted(set(boundaries + [len(sentences)]))

    for i in range(len(all_boundaries) - 1):
        start_idx = all_boundaries[i]
        end_idx = all_boundaries[i + 1]

        segment_sentences = sentences[start_idx:end_idx]
        segment_text = " ".join(segment_sentences)

        # Extract keywords
        keywords = list(_extract_keywords(segment_text))[:10]

        # Generate title from keywords and first sentence
        title = _generate_topic_title(segment_sentences, keywords)

        # Generate summary
        summary = _generate_topic_summary(segment_sentences)

        # Compute timestamps
        start_time, end_time = _estimate_timestamps(
            start_idx, end_idx, len(sentences), timed_segments
        )

        topics.append(TopicSegment(
            topic_id=i,
            title=title,
            summary=summary,
            start_time=start_time,
            end_time=end_time,
            start_index=start_idx,
            end_index=end_idx,
            text=segment_text,
            keywords=keywords,
            confidence=0.7 if len(segment_sentences) >= 5 else 0.5,
        ))

    return topics


def _single_topic(
    transcript: str,
    sentences: List[str],
    segments: Optional[List[Dict]],
) -> TopicSegment:
    """Create a single topic for short transcripts."""
    duration = 0.0
    if segments:
        duration = max(s.get("end", 0) for s in segments) if segments else 0.0

    return TopicSegment(
        topic_id=0,
        title="Main Content",
        summary=transcript[:300] + "..." if len(transcript) > 300 else transcript,
        start_time=0.0,
        end_time=duration,
        start_index=0,
        end_index=len(sentences),
        text=transcript,
        keywords=list(_extract_keywords(transcript))[:10],
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Automated chapter detection
# ---------------------------------------------------------------------------

def detect_chapters(
    topics: List[TopicSegment],
    min_chapter_duration: float = 60.0,
    max_chapters: int = 15,
    merge_threshold: float = 0.6,
) -> List[Chapter]:
    """
    Auto-detect chapters by merging related topic segments.

    Chapters are higher-level groupings of topics. Short or closely
    related topics get merged into single chapters.

    Args:
        topics: List of detected topic segments
        min_chapter_duration: Minimum chapter duration in seconds
        max_chapters: Maximum number of chapters
        merge_threshold: Keyword overlap threshold for merging topics

    Returns:
        List of Chapter objects
    """
    if not topics:
        return []

    if len(topics) == 1:
        topic = topics[0]
        return [Chapter(
            chapter_id=0,
            title=topic.title,
            start_time=topic.start_time,
            end_time=topic.end_time,
            duration=topic.end_time - topic.start_time,
            summary=topic.summary,
            topic_ids=[topic.topic_id],
        )]

    # Group related topics into chapters
    chapter_groups: List[List[TopicSegment]] = []
    current_group: List[TopicSegment] = [topics[0]]

    for i in range(1, len(topics)):
        prev_topic = topics[i - 1]
        curr_topic = topics[i]

        # Check if topics should be merged
        should_merge = False

        # Merge if combined duration is still short
        group_duration = sum(t.end_time - t.start_time for t in current_group)
        if group_duration < min_chapter_duration:
            should_merge = True

        # Merge if keywords overlap significantly
        if not should_merge:
            prev_kw = set(prev_topic.keywords)
            curr_kw = set(curr_topic.keywords)
            if prev_kw and curr_kw:
                overlap = len(prev_kw & curr_kw) / max(len(prev_kw | curr_kw), 1)
                if overlap > merge_threshold:
                    should_merge = True

        if should_merge:
            current_group.append(curr_topic)
        else:
            chapter_groups.append(current_group)
            current_group = [curr_topic]

    chapter_groups.append(current_group)

    # Limit chapters
    while len(chapter_groups) > max_chapters and len(chapter_groups) > 1:
        # Merge the two shortest adjacent groups
        min_duration = float("inf")
        merge_idx = 0
        for i in range(len(chapter_groups) - 1):
            combined_duration = sum(
                t.end_time - t.start_time
                for t in chapter_groups[i] + chapter_groups[i + 1]
            )
            if combined_duration < min_duration:
                min_duration = combined_duration
                merge_idx = i
        chapter_groups[merge_idx].extend(chapter_groups[merge_idx + 1])
        del chapter_groups[merge_idx + 1]

    # Build Chapter objects
    chapters: List[Chapter] = []
    for idx, group in enumerate(chapter_groups):
        start_time = group[0].start_time
        end_time = group[-1].end_time

        # Generate chapter title from primary topic
        primary_topic = max(group, key=lambda t: len(t.text))
        title = primary_topic.title

        # Generate summary from all topics in chapter
        summary_parts = [t.summary for t in group if t.summary]
        summary = " ".join(summary_parts)[:500]

        chapters.append(Chapter(
            chapter_id=idx,
            title=title,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            summary=summary,
            topic_ids=[t.topic_id for t in group],
        ))

    logger.info(f"Detected {len(chapters)} chapters from {len(topics)} topics")
    return chapters


def generate_youtube_chapters(chapters: List[Chapter]) -> str:
    """
    Generate YouTube-compatible chapter timestamps.

    Format:
    0:00 Introduction
    2:15 Topic One
    5:30 Topic Two
    ...
    """
    lines = []
    for chapter in chapters:
        timestamp = _format_timestamp(chapter.start_time)
        lines.append(f"{timestamp} {chapter.title}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full segmentation pipeline
# ---------------------------------------------------------------------------

def segment_content(
    transcript: str,
    segments: Optional[List[Dict]] = None,
    min_topic_sentences: int = 3,
    max_topics: int = 20,
    min_chapter_duration: float = 60.0,
    max_chapters: int = 15,
) -> SegmentationResult:
    """
    Complete content segmentation pipeline.

    Args:
        transcript: Full transcript text
        segments: Optional timed segments
        min_topic_sentences: Minimum sentences per topic
        max_topics: Maximum topics
        min_chapter_duration: Minimum chapter duration (seconds)
        max_chapters: Maximum chapters

    Returns:
        SegmentationResult with topics and chapters
    """
    logger.info("Starting content segmentation")

    # Step 1: Topic segmentation
    topics = segment_topics(
        transcript,
        segments=segments,
        min_topic_sentences=min_topic_sentences,
        max_topics=max_topics,
    )

    # Step 2: Chapter detection
    chapters = detect_chapters(
        topics,
        min_chapter_duration=min_chapter_duration,
        max_chapters=max_chapters,
    )

    # Compute metrics
    topic_durations = [t.end_time - t.start_time for t in topics if t.end_time > t.start_time]
    avg_duration = sum(topic_durations) / max(len(topic_durations), 1)

    content_flow = [t.title for t in topics]

    return SegmentationResult(
        topics=topics,
        chapters=chapters,
        total_topics=len(topics),
        total_chapters=len(chapters),
        avg_topic_duration=avg_duration,
        content_flow=content_flow,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Handles English, Hindi, and Sanskrit sentence boundaries.
    """
    # Split on sentence-ending punctuation
    # Include Devanagari danda (।) and double danda (॥)
    pattern = r'(?<=[.!?।॥])\s+'
    sentences = re.split(pattern, text)

    # Filter empty and very short sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    # If no sentence boundaries found, split on newlines or fixed chunks
    if len(sentences) <= 1 and len(text) > 100:
        # Try newline splitting
        sentences = [s.strip() for s in text.split("\n") if s.strip() and len(s.strip()) > 10]

    if len(sentences) <= 1 and len(text) > 200:
        # Fixed-size chunks as last resort
        words = text.split()
        chunk_size = 30  # words per chunk
        sentences = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                sentences.append(chunk.strip())

    return sentences


def _extract_keywords(text: str, max_keywords: int = 15) -> set:
    """Extract keywords from text using simple TF-based scoring."""
    # Common stop words (English + Hindi)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "about", "this", "that", "these", "those", "it",
        "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "his", "she", "her", "they", "them", "their", "what", "which", "who",
        "whom", "up", "also", "like", "know", "think", "say", "said",
        # Hindi stop words
        "है", "हैं", "था", "थी", "थे", "को", "का", "की", "के", "में",
        "से", "पर", "ने", "और", "या", "भी", "तो", "ही", "पर", "जो",
        "कि", "यह", "वह", "इस", "उस", "एक", "नहीं", "कर", "हो",
    }

    # Tokenize and filter
    words = re.findall(r'\b[\w\u0900-\u097F]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]

    # Count frequencies
    freq: Dict[str, int] = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    # Return top keywords
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return {w for w, _ in sorted_words[:max_keywords]}


def _generate_topic_title(sentences: List[str], keywords: List[str]) -> str:
    """Generate a concise topic title from sentences and keywords."""
    if not sentences:
        return "Untitled Topic"

    # Use first sentence as basis, truncated
    first = sentences[0]

    # If first sentence is short enough, use it
    if len(first) <= 60:
        return first.rstrip(".")

    # Otherwise, use top keywords
    if keywords:
        title_words = keywords[:4]
        return " — ".join(w.title() for w in title_words)

    # Truncate first sentence
    words = first.split()[:8]
    return " ".join(words) + "..."


def _generate_topic_summary(sentences: List[str]) -> str:
    """Generate a brief summary from topic sentences."""
    if not sentences:
        return ""

    # Use first 2-3 sentences as summary
    summary_sentences = sentences[:3]
    summary = " ".join(summary_sentences)

    if len(summary) > 400:
        summary = summary[:397] + "..."

    return summary


def _estimate_timestamps(
    start_idx: int,
    end_idx: int,
    total_sentences: int,
    timed_segments: Optional[List[Dict]],
) -> Tuple[float, float]:
    """Estimate timestamps for a sentence range."""
    if timed_segments and len(timed_segments) > 0:
        total_duration = max(s.get("end", 0) for s in timed_segments)
        # Linear interpolation based on sentence position
        start_time = (start_idx / max(total_sentences, 1)) * total_duration
        end_time = (end_idx / max(total_sentences, 1)) * total_duration
        return start_time, end_time

    # Rough estimate: ~3 seconds per sentence (average speaking rate)
    return start_idx * 3.0, end_idx * 3.0


def _format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
