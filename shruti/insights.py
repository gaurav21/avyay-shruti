"""
Key Insights Extraction module for ŚRUTI V3.0.
Detects important moments, action items, sentiment shifts, and quotable highlights
from video transcripts using multi-signal analysis.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class InsightType(str, Enum):
    KEY_POINT = "key_point"
    ACTION_ITEM = "action_item"
    QUESTION = "question"
    QUOTE = "quote"
    DEFINITION = "definition"
    STATISTIC = "statistic"
    EMPHASIS = "emphasis"
    TRANSITION = "transition"
    CONCLUSION = "conclusion"


class SentimentLabel(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class Insight:
    """A single extracted insight from the video."""
    insight_id: str
    insight_type: InsightType
    text: str
    start_time: float
    end_time: float
    confidence: float
    importance_score: float  # 0.0 - 1.0
    context: str = ""  # surrounding context
    speaker_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentSegment:
    """Sentiment analysis for a segment of the video."""
    start_time: float
    end_time: float
    label: SentimentLabel
    score: float  # -1.0 to 1.0
    text: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class ActionItem:
    """An identified action item from the video."""
    action_id: str
    text: str
    assignee: Optional[str] = None
    priority: str = "medium"  # low, medium, high
    start_time: float = 0.0
    context: str = ""
    confidence: float = 0.0


@dataclass
class HighlightReel:
    """A collection of highlights for a video."""
    video_id: str
    total_duration: float
    highlights: List[Insight]
    top_quotes: List[Insight]
    action_items: List[ActionItem]
    sentiment_arc: List[SentimentSegment]
    summary_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightsResult:
    """Complete insights extraction result."""
    insights: List[Insight]
    action_items: List[ActionItem]
    sentiment_segments: List[SentimentSegment]
    highlight_reel: HighlightReel
    total_insights: int
    insight_type_counts: Dict[str, int] = field(default_factory=dict)
    overall_sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    avg_sentiment_score: float = 0.0
    quality_score: float = 0.0


# ---------------------------------------------------------------------------
# Insight extraction patterns
# ---------------------------------------------------------------------------

# Patterns that indicate emphasis or importance
EMPHASIS_PATTERNS = [
    r"\b(?:importantly|crucially|key\s+point|essential|critical|fundamental)\b",
    r"\b(?:remember|note\s+that|pay\s+attention|keep\s+in\s+mind)\b",
    r"\b(?:the\s+most\s+important|the\s+key\s+takeaway|the\s+main\s+point)\b",
    r"\b(?:above\s+all|first\s+and\s+foremost|most\s+notably)\b",
]

# Patterns for action items
ACTION_PATTERNS = [
    r"\b(?:you\s+should|you\s+need\s+to|make\s+sure|don'?t\s+forget)\b",
    r"\b(?:take\s+action|next\s+step|todo|to-do|action\s+item)\b",
    r"\b(?:go\s+ahead\s+and|try\s+to|start\s+by|begin\s+with)\b",
    r"\b(?:implement|deploy|configure|set\s+up|install)\b",
    r"\b(?:always\s+(?:make\s+sure|ensure|check|verify))\b",
]

# Patterns for questions (rhetorical or real)
QUESTION_PATTERNS = [
    r"[^.!]*\?\s*$",
    r"\b(?:what\s+if|how\s+do|why\s+does|when\s+should|where\s+can)\b",
    r"\b(?:have\s+you\s+ever|did\s+you\s+know|can\s+you\s+imagine)\b",
]

# Patterns for definitions
DEFINITION_PATTERNS = [
    r"\b(?:is\s+defined\s+as|means\s+that|refers\s+to|is\s+known\s+as)\b",
    r"\b(?:in\s+other\s+words|that\s+is\s+to\s+say|simply\s+put)\b",
    r"\b(?:the\s+definition\s+of|what\s+we\s+mean\s+by)\b",
]

# Patterns for statistics / data points
STATISTIC_PATTERNS = [
    r"\b\d+(?:\.\d+)?%\b",
    r"\b(?:according\s+to|research\s+shows|studies\s+show|data\s+suggests)\b",
    r"\b(?:statistically|on\s+average|the\s+majority|most\s+of)\b",
    r"\b\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|k|m|b))?\b",
]

# Patterns for conclusions
CONCLUSION_PATTERNS = [
    r"\b(?:in\s+conclusion|to\s+summarize|to\s+sum\s+up|in\s+summary)\b",
    r"\b(?:finally|lastly|to\s+wrap\s+up|the\s+bottom\s+line)\b",
    r"\b(?:so\s+in\s+short|all\s+in\s+all|overall)\b",
]

# Transition phrases
TRANSITION_PATTERNS = [
    r"\b(?:now\s+let'?s\s+(?:talk|move|look|discuss|turn))\b",
    r"\b(?:moving\s+on\s+to|next\s+up|speaking\s+of)\b",
    r"\b(?:on\s+another\s+note|shifting\s+gears|let'?s\s+switch)\b",
]

# Simple positive/negative sentiment words
POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "good", "best",
    "love", "awesome", "brilliant", "perfect", "beautiful", "impressive",
    "powerful", "remarkable", "outstanding", "incredible", "success",
    "benefit", "advantage", "improve", "achieve", "win", "gain", "grow",
    "happy", "exciting", "innovative", "elegant", "efficient", "robust",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "failure", "problem", "issue", "bug", "error", "crash", "broken",
    "difficult", "complex", "confusing", "frustrating", "annoying",
    "slow", "expensive", "vulnerable", "dangerous", "risk", "threat",
    "loss", "decline", "decrease", "unfortunately", "sadly", "worse",
}


# ---------------------------------------------------------------------------
# Core insight extraction
# ---------------------------------------------------------------------------

def extract_insights(
    transcript: str,
    segments: Optional[List[Dict]] = None,
    video_id: str = "",
    title: str = "",
    speaker_segments: Optional[List[Dict]] = None,
    max_insights: int = 50,
    min_importance: float = 0.3,
) -> InsightsResult:
    """
    Extract key insights from a video transcript.

    Analyzes the transcript for:
    - Key points and emphasis moments
    - Action items and recommendations
    - Questions (rhetorical and real)
    - Definitions and explanations
    - Statistics and data points
    - Quotable highlights
    - Sentiment arc across the video

    Args:
        transcript: Full transcript text
        segments: Optional timed segments [{start, end, text}, ...]
        video_id: Video identifier
        title: Video title for context
        speaker_segments: Optional speaker-attributed segments
        max_insights: Maximum insights to return
        min_importance: Minimum importance score threshold

    Returns:
        InsightsResult with all extracted insights
    """
    logger.info(f"Extracting insights from transcript ({len(transcript)} chars)")

    # Split transcript into sentences with timing
    timed_sentences = _build_timed_sentences(transcript, segments)

    # Extract different types of insights
    all_insights: List[Insight] = []
    insight_counter = 0

    for sent_idx, (text, start, end) in enumerate(timed_sentences):
        context_before = timed_sentences[max(0, sent_idx - 1)][0] if sent_idx > 0 else ""
        context_after = timed_sentences[min(len(timed_sentences) - 1, sent_idx + 1)][0] if sent_idx < len(timed_sentences) - 1 else ""
        context = f"{context_before} [...] {context_after}"

        # Check each pattern type
        for insight_type, patterns in [
            (InsightType.EMPHASIS, EMPHASIS_PATTERNS),
            (InsightType.ACTION_ITEM, ACTION_PATTERNS),
            (InsightType.QUESTION, QUESTION_PATTERNS),
            (InsightType.DEFINITION, DEFINITION_PATTERNS),
            (InsightType.STATISTIC, STATISTIC_PATTERNS),
            (InsightType.CONCLUSION, CONCLUSION_PATTERNS),
            (InsightType.TRANSITION, TRANSITION_PATTERNS),
        ]:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    importance = _calculate_importance(
                        text, insight_type, sent_idx, len(timed_sentences)
                    )
                    if importance >= min_importance:
                        insight_counter += 1
                        # Find speaker if available
                        speaker = _find_speaker(start, speaker_segments)

                        all_insights.append(Insight(
                            insight_id=f"INS-{video_id[:8]}-{insight_counter:04d}",
                            insight_type=insight_type,
                            text=text.strip(),
                            start_time=start,
                            end_time=end,
                            confidence=_pattern_confidence(text, patterns),
                            importance_score=importance,
                            context=context[:300],
                            speaker_id=speaker,
                            tags=_extract_insight_tags(text, insight_type),
                        ))
                    break  # Only match first pattern per type per sentence

    # Detect quotes (standalone significant statements)
    quotes = _extract_quotes(timed_sentences, video_id, speaker_segments)
    all_insights.extend(quotes)

    # Detect repeated phrases (emphasis through repetition)
    repetitions = _detect_repetition_emphasis(timed_sentences, video_id)
    all_insights.extend(repetitions)

    # Sort by importance and deduplicate
    all_insights = _deduplicate_insights(all_insights)
    all_insights.sort(key=lambda i: i.importance_score, reverse=True)
    all_insights = all_insights[:max_insights]

    # Extract action items
    action_items = _extract_action_items(all_insights, timed_sentences)

    # Compute sentiment arc
    sentiment_segments = _compute_sentiment_arc(timed_sentences)

    # Build highlight reel
    highlight_reel = _build_highlight_reel(
        video_id=video_id,
        total_duration=timed_sentences[-1][2] if timed_sentences else 0.0,
        insights=all_insights,
        action_items=action_items,
        sentiment_segments=sentiment_segments,
    )

    # Compute type counts
    type_counts: Dict[str, int] = {}
    for ins in all_insights:
        type_counts[ins.insight_type.value] = type_counts.get(ins.insight_type.value, 0) + 1

    # Overall sentiment
    overall_sentiment, avg_score = _compute_overall_sentiment(sentiment_segments)

    # Quality score
    quality = _compute_insights_quality(all_insights, action_items, sentiment_segments, transcript)

    logger.info(
        f"Extracted {len(all_insights)} insights, {len(action_items)} action items, "
        f"{len(sentiment_segments)} sentiment segments"
    )

    return InsightsResult(
        insights=all_insights,
        action_items=action_items,
        sentiment_segments=sentiment_segments,
        highlight_reel=highlight_reel,
        total_insights=len(all_insights),
        insight_type_counts=type_counts,
        overall_sentiment=overall_sentiment,
        avg_sentiment_score=avg_score,
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

def _compute_sentiment_arc(
    timed_sentences: List[Tuple[str, float, float]],
    window_size: int = 5,
) -> List[SentimentSegment]:
    """
    Compute sentiment arc across the video using sliding windows.
    Uses a lexicon-based approach (no ML dependencies required).
    """
    if not timed_sentences:
        return []

    segments: List[SentimentSegment] = []

    # Process in windows
    for i in range(0, len(timed_sentences), max(1, window_size // 2)):
        window = timed_sentences[i:i + window_size]
        if not window:
            break

        combined_text = " ".join(s[0] for s in window)
        words = set(re.findall(r'\b\w+\b', combined_text.lower()))

        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total

        # Classify
        if score > 0.5:
            label = SentimentLabel.VERY_POSITIVE
        elif score > 0.15:
            label = SentimentLabel.POSITIVE
        elif score < -0.5:
            label = SentimentLabel.VERY_NEGATIVE
        elif score < -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        # Extract sentiment keywords
        sentiment_kw = list((words & POSITIVE_WORDS) | (words & NEGATIVE_WORDS))

        segments.append(SentimentSegment(
            start_time=window[0][1],
            end_time=window[-1][2],
            label=label,
            score=score,
            text=combined_text[:200],
            keywords=sentiment_kw[:10],
        ))

    return segments


def _compute_overall_sentiment(
    segments: List[SentimentSegment],
) -> Tuple[SentimentLabel, float]:
    """Compute overall sentiment from segment scores."""
    if not segments:
        return SentimentLabel.NEUTRAL, 0.0

    avg_score = sum(s.score for s in segments) / len(segments)

    if avg_score > 0.3:
        label = SentimentLabel.VERY_POSITIVE
    elif avg_score > 0.1:
        label = SentimentLabel.POSITIVE
    elif avg_score < -0.3:
        label = SentimentLabel.VERY_NEGATIVE
    elif avg_score < -0.1:
        label = SentimentLabel.NEGATIVE
    else:
        label = SentimentLabel.NEUTRAL

    return label, avg_score


# ---------------------------------------------------------------------------
# Quote extraction
# ---------------------------------------------------------------------------

def _extract_quotes(
    timed_sentences: List[Tuple[str, float, float]],
    video_id: str,
    speaker_segments: Optional[List[Dict]] = None,
    min_length: int = 30,
    max_length: int = 200,
) -> List[Insight]:
    """
    Extract quotable, impactful statements from the transcript.
    Uses heuristics: sentence length, structure, rhetorical devices.
    """
    quotes: List[Insight] = []
    counter = 0

    for idx, (text, start, end) in enumerate(timed_sentences):
        text_stripped = text.strip()
        if len(text_stripped) < min_length or len(text_stripped) > max_length:
            continue

        score = 0.0

        # Self-contained statement (not referencing "this", "that", etc.)
        if not re.search(r'\b(?:this|that|these|those|it)\s+(?:is|was|are|were)\b', text_stripped, re.IGNORECASE):
            score += 0.1

        # Contains strong language
        strong_words = {"never", "always", "every", "must", "truth", "believe", "power", "change", "future"}
        words_lower = set(text_stripped.lower().split())
        if words_lower & strong_words:
            score += 0.2

        # Metaphor/simile patterns
        if re.search(r'\b(?:like\s+a|as\s+if|just\s+as)\b', text_stripped, re.IGNORECASE):
            score += 0.15

        # Parallelism (repeated structure)
        if re.search(r'(\w+)\s+.*\1', text_stripped):
            score += 0.1

        # Short, punchy (< 80 chars)
        if len(text_stripped) < 80:
            score += 0.1

        # Contains quotation marks (explicit quote)
        if '"' in text_stripped or '"' in text_stripped or '«' in text_stripped:
            score += 0.3

        if score >= 0.3:
            counter += 1
            speaker = _find_speaker(start, speaker_segments)
            quotes.append(Insight(
                insight_id=f"QUO-{video_id[:8]}-{counter:04d}",
                insight_type=InsightType.QUOTE,
                text=text_stripped,
                start_time=start,
                end_time=end,
                confidence=min(1.0, score),
                importance_score=min(1.0, score + 0.1),
                speaker_id=speaker,
                tags=["quote", "highlight"],
            ))

    return quotes


# ---------------------------------------------------------------------------
# Repetition emphasis detection
# ---------------------------------------------------------------------------

def _detect_repetition_emphasis(
    timed_sentences: List[Tuple[str, float, float]],
    video_id: str,
    min_phrase_length: int = 3,
    min_occurrences: int = 2,
) -> List[Insight]:
    """
    Detect phrases repeated multiple times (emphasis through repetition).
    Speakers often repeat key phrases to drive points home.
    """
    insights: List[Insight] = []

    # Extract n-grams (3-6 words) and count occurrences
    phrase_occurrences: Dict[str, List[int]] = {}

    for idx, (text, _, _) in enumerate(timed_sentences):
        words = text.lower().split()
        for n in range(min_phrase_length, min(7, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                # Skip if too common / stopword-heavy
                if _is_stopword_heavy(phrase):
                    continue
                if phrase not in phrase_occurrences:
                    phrase_occurrences[phrase] = []
                phrase_occurrences[phrase].append(idx)

    # Find repeated phrases
    counter = 0
    seen_phrases = set()

    for phrase, indices in sorted(
        phrase_occurrences.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    ):
        if len(indices) < min_occurrences:
            continue

        # Deduplicate overlapping phrases
        if any(phrase in sp or sp in phrase for sp in seen_phrases):
            continue
        seen_phrases.add(phrase)

        first_idx = indices[0]
        text, start, end = timed_sentences[first_idx]

        counter += 1
        importance = min(1.0, 0.4 + 0.1 * len(indices))

        insights.append(Insight(
            insight_id=f"REP-{video_id[:8]}-{counter:04d}",
            insight_type=InsightType.EMPHASIS,
            text=phrase,
            start_time=start,
            end_time=end,
            confidence=0.7,
            importance_score=importance,
            context=f"Repeated {len(indices)} times across the video",
            tags=["repetition", "emphasis"],
            metadata={"occurrences": len(indices), "indices": indices[:10]},
        ))

        if counter >= 10:
            break

    return insights


# ---------------------------------------------------------------------------
# Action item extraction
# ---------------------------------------------------------------------------

def _extract_action_items(
    insights: List[Insight],
    timed_sentences: List[Tuple[str, float, float]],
) -> List[ActionItem]:
    """Extract structured action items from insights and transcript."""
    items: List[ActionItem] = []
    counter = 0

    # From insights tagged as action items
    for ins in insights:
        if ins.insight_type == InsightType.ACTION_ITEM:
            counter += 1
            priority = "high" if ins.importance_score > 0.7 else "medium" if ins.importance_score > 0.4 else "low"
            items.append(ActionItem(
                action_id=f"ACT-{counter:04d}",
                text=ins.text,
                priority=priority,
                start_time=ins.start_time,
                context=ins.context,
                confidence=ins.confidence,
            ))

    # Additional extraction from imperative sentences
    for text, start, end in timed_sentences:
        text_stripped = text.strip()
        # Imperative mood detection
        if re.match(r'^(?:Make\s+sure|Ensure|Always|Never|Don\'t\s+forget|Remember\s+to)\b', text_stripped):
            counter += 1
            items.append(ActionItem(
                action_id=f"ACT-{counter:04d}",
                text=text_stripped,
                priority="medium",
                start_time=start,
                confidence=0.6,
            ))

    # Deduplicate
    seen_texts = set()
    unique_items = []
    for item in items:
        key = item.text.lower()[:60]
        if key not in seen_texts:
            seen_texts.add(key)
            unique_items.append(item)

    return unique_items


# ---------------------------------------------------------------------------
# Highlight reel builder
# ---------------------------------------------------------------------------

def _build_highlight_reel(
    video_id: str,
    total_duration: float,
    insights: List[Insight],
    action_items: List[ActionItem],
    sentiment_segments: List[SentimentSegment],
) -> HighlightReel:
    """Build a highlight reel from all extracted data."""
    # Top highlights by importance
    highlights = sorted(insights, key=lambda i: i.importance_score, reverse=True)[:20]

    # Top quotes
    quotes = [i for i in insights if i.insight_type == InsightType.QUOTE]
    top_quotes = sorted(quotes, key=lambda i: i.importance_score, reverse=True)[:10]

    # Summary stats
    type_dist = Counter(i.insight_type.value for i in insights)
    sentiment_dist = Counter(s.label.value for s in sentiment_segments)

    stats = {
        "total_insights": len(insights),
        "total_action_items": len(action_items),
        "insight_types": dict(type_dist),
        "sentiment_distribution": dict(sentiment_dist),
        "highlight_density": len(highlights) / max(total_duration / 60, 1),  # per minute
        "avg_importance": sum(i.importance_score for i in insights) / max(len(insights), 1),
    }

    return HighlightReel(
        video_id=video_id,
        total_duration=total_duration,
        highlights=highlights,
        top_quotes=top_quotes,
        action_items=action_items,
        sentiment_arc=sentiment_segments,
        summary_stats=stats,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_timed_sentences(
    transcript: str,
    segments: Optional[List[Dict]] = None,
) -> List[Tuple[str, float, float]]:
    """Build list of (text, start_time, end_time) tuples."""
    if segments:
        result = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                result.append((
                    text,
                    float(seg.get("start", 0)),
                    float(seg.get("end", 0)),
                ))
        if result:
            return result

    # Fallback: split by sentences and estimate timing
    pattern = r'(?<=[.!?।॥])\s+'
    sentences = re.split(pattern, transcript)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if not sentences:
        if transcript.strip():
            return [(transcript.strip(), 0.0, len(transcript.split()) * 0.4)]
        return []

    # Estimate ~3s per sentence (average speaking rate)
    result = []
    current_time = 0.0
    for sent in sentences:
        duration = max(1.0, len(sent.split()) * 0.4)  # ~0.4s per word
        result.append((sent, current_time, current_time + duration))
        current_time += duration

    return result


def _calculate_importance(
    text: str,
    insight_type: InsightType,
    position_idx: int,
    total_sentences: int,
) -> float:
    """Calculate importance score for an insight."""
    score = 0.0

    # Base score by type
    type_weights = {
        InsightType.KEY_POINT: 0.7,
        InsightType.ACTION_ITEM: 0.6,
        InsightType.STATISTIC: 0.6,
        InsightType.DEFINITION: 0.5,
        InsightType.CONCLUSION: 0.7,
        InsightType.EMPHASIS: 0.6,
        InsightType.QUESTION: 0.4,
        InsightType.QUOTE: 0.5,
        InsightType.TRANSITION: 0.2,
    }
    score = type_weights.get(insight_type, 0.3)

    # Position bonus (beginning and end are more important)
    rel_pos = position_idx / max(total_sentences - 1, 1)
    if rel_pos < 0.15 or rel_pos > 0.85:
        score += 0.15
    elif rel_pos < 0.3 or rel_pos > 0.7:
        score += 0.05

    # Length bonus (concise insights are often more impactful)
    word_count = len(text.split())
    if 8 <= word_count <= 25:
        score += 0.1

    # Specificity bonus (contains numbers, names, technical terms)
    if re.search(r'\d+', text):
        score += 0.05
    if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text):  # Proper nouns
        score += 0.05

    return min(1.0, score)


def _pattern_confidence(text: str, patterns: List[str]) -> float:
    """Calculate confidence based on pattern match quality."""
    matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return min(1.0, 0.5 + 0.15 * matches)


def _find_speaker(
    timestamp: float,
    speaker_segments: Optional[List[Dict]],
) -> Optional[str]:
    """Find the speaker at a given timestamp."""
    if not speaker_segments:
        return None
    for seg in speaker_segments:
        if seg.get("start", 0) <= timestamp <= seg.get("end", 0):
            return seg.get("speaker_id")
    return None


def _extract_insight_tags(text: str, insight_type: InsightType) -> List[str]:
    """Extract relevant tags for an insight."""
    tags = [insight_type.value]

    # Technical terms
    tech_terms = re.findall(r'\b(?:API|SDK|AI|ML|UI|UX|DevOps|CI/CD|REST|GraphQL|SQL|NoSQL)\b', text, re.IGNORECASE)
    tags.extend(t.upper() for t in tech_terms)

    # Domain detection
    if re.search(r'\b(?:kubernetes|docker|cloud|server|deploy)\b', text, re.IGNORECASE):
        tags.append("infrastructure")
    if re.search(r'\b(?:security|auth|encrypt|vulnerability|CVE)\b', text, re.IGNORECASE):
        tags.append("security")
    if re.search(r'\b(?:performance|latency|throughput|cache|optimize)\b', text, re.IGNORECASE):
        tags.append("performance")

    return list(set(tags))[:10]


def _is_stopword_heavy(phrase: str) -> bool:
    """Check if a phrase is mostly stopwords."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "this", "that", "it", "we", "you",
        "he", "she", "they", "i", "me", "my", "so", "if", "as",
    }
    words = phrase.split()
    stop_count = sum(1 for w in words if w in stopwords)
    return stop_count / max(len(words), 1) > 0.6


def _deduplicate_insights(insights: List[Insight]) -> List[Insight]:
    """Remove duplicate or near-duplicate insights."""
    if not insights:
        return []

    unique: List[Insight] = []
    seen_texts: set = set()

    for ins in insights:
        key = ins.text.lower()[:80]
        if key not in seen_texts:
            seen_texts.add(key)
            unique.append(ins)

    return unique


def _compute_insights_quality(
    insights: List[Insight],
    action_items: List[ActionItem],
    sentiment_segments: List[SentimentSegment],
    transcript: str,
) -> float:
    """Compute a quality score (0-1) for the extracted insights."""
    if not transcript:
        return 0.0

    score = 0.0
    max_score = 5.0

    # Diversity of insight types (max 1.0)
    types_found = len(set(i.insight_type for i in insights))
    score += min(1.0, types_found / 5.0)

    # Reasonable density (max 1.0)
    words = len(transcript.split())
    density = len(insights) / max(words / 100, 1)  # insights per 100 words
    if 0.5 <= density <= 5.0:
        score += 1.0
    elif density > 0.1:
        score += 0.5

    # Has action items (max 1.0)
    if action_items:
        score += min(1.0, len(action_items) / 3)

    # Has sentiment coverage (max 1.0)
    if sentiment_segments:
        score += min(1.0, len(sentiment_segments) / 5)

    # Average confidence (max 1.0)
    if insights:
        avg_conf = sum(i.confidence for i in insights) / len(insights)
        score += avg_conf

    return min(1.0, score / max_score)
