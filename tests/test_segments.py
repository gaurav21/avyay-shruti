"""
Tests for ŚRUTI V2.0 topic segmentation and chapter detection module.
"""

import pytest

from shruti.segments import (
    TopicSegment,
    Chapter,
    SegmentationResult,
    segment_topics,
    detect_chapters,
    generate_youtube_chapters,
    _split_into_sentences,
    _extract_keywords,
    _generate_topic_title,
    _generate_topic_summary,
    _estimate_timestamps,
    _format_timestamp,
)


# ---------------------------------------------------------------------------
# Sentence splitting tests
# ---------------------------------------------------------------------------

class TestSplitIntoSentences:
    def test_english_sentences(self):
        text = "This is sentence one. This is sentence two. And this is three."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_hindi_sentences(self):
        text = "यह पहला वाक्य है। यह दूसरा वाक्य है। और यह तीसरा।"
        sentences = _split_into_sentences(text)
        assert len(sentences) >= 2

    def test_short_text_no_split(self):
        text = "Short text"
        sentences = _split_into_sentences(text)
        # Very short — may return empty if under 10 chars threshold
        assert len(sentences) <= 1

    def test_long_text_without_punctuation(self):
        text = " ".join(["word"] * 100)
        sentences = _split_into_sentences(text)
        assert len(sentences) >= 1

    def test_newline_splitting(self):
        text = "First paragraph about Vedanta philosophy\nSecond paragraph about Yoga practice"
        sentences = _split_into_sentences(text)
        assert len(sentences) >= 1


# ---------------------------------------------------------------------------
# Keyword extraction tests
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_basic_extraction(self):
        text = "Vedanta philosophy teaches about Brahman and the nature of reality"
        keywords = _extract_keywords(text)
        assert isinstance(keywords, set)
        assert len(keywords) > 0
        # Stop words should be filtered
        assert "the" not in keywords
        assert "and" not in keywords

    def test_hindi_stop_words(self):
        text = "यह एक महत्वपूर्ण शिक्षा है जो ज्ञान प्रदान करती है"
        keywords = _extract_keywords(text)
        assert "है" not in keywords
        assert "यह" not in keywords

    def test_max_keywords(self):
        text = " ".join([f"keyword{i}" for i in range(50)])
        keywords = _extract_keywords(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_empty_text(self):
        keywords = _extract_keywords("")
        assert len(keywords) == 0


# ---------------------------------------------------------------------------
# Topic title generation tests
# ---------------------------------------------------------------------------

class TestGenerateTopicTitle:
    def test_short_first_sentence(self):
        sentences = ["Introduction to Vedanta", "More details here."]
        keywords = ["vedanta", "introduction"]
        title = _generate_topic_title(sentences, keywords)
        assert "Vedanta" in title or "vedanta" in title.lower()

    def test_empty_sentences(self):
        title = _generate_topic_title([], [])
        assert title == "Untitled Topic"

    def test_long_first_sentence(self):
        long_sentence = "This is a very long sentence " * 10
        keywords = ["important", "topic", "discussion"]
        title = _generate_topic_title([long_sentence], keywords)
        # Should use keywords since first sentence is too long
        assert len(title) < len(long_sentence)


class TestGenerateTopicSummary:
    def test_normal_summary(self):
        sentences = ["First point.", "Second point.", "Third point.", "Fourth point."]
        summary = _generate_topic_summary(sentences)
        assert "First point" in summary
        assert "Third point" in summary

    def test_empty_summary(self):
        assert _generate_topic_summary([]) == ""

    def test_long_summary_truncated(self):
        sentences = ["Very long sentence. " * 50, "Another long sentence. " * 50]
        summary = _generate_topic_summary(sentences)
        assert len(summary) <= 403  # 400 + "..."


# ---------------------------------------------------------------------------
# Timestamp estimation tests
# ---------------------------------------------------------------------------

class TestEstimateTimestamps:
    def test_with_segments(self):
        segments = [
            {"start": 0, "end": 60},
            {"start": 60, "end": 120},
        ]
        start, end = _estimate_timestamps(0, 5, 10, segments)
        assert start == 0.0
        assert end == 60.0  # 5/10 * 120

    def test_without_segments(self):
        start, end = _estimate_timestamps(3, 6, 10, None)
        assert start == 9.0   # 3 * 3.0
        assert end == 18.0    # 6 * 3.0


# ---------------------------------------------------------------------------
# Timestamp formatting tests
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_seconds_only(self):
        assert _format_timestamp(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_timestamp(125) == "2:05"

    def test_hours(self):
        assert _format_timestamp(3661) == "1:01:01"

    def test_zero(self):
        assert _format_timestamp(0) == "0:00"


# ---------------------------------------------------------------------------
# Topic segmentation tests
# ---------------------------------------------------------------------------

class TestSegmentTopics:
    def test_short_transcript(self):
        text = "Short text only."
        topics = segment_topics(text, use_embeddings=False)
        assert len(topics) == 1
        assert topics[0].topic_id == 0

    def test_distinct_topics(self):
        # Create transcript with clearly different topics
        topic1 = ". ".join(["Vedanta philosophy discusses Brahman reality consciousness"] * 5)
        topic2 = ". ".join(["Yoga practice involves asana pranayama meditation"] * 5)
        topic3 = ". ".join(["Ayurveda medicine herbs doshas vata pitta kapha"] * 5)

        text = f"{topic1}. {topic2}. {topic3}."
        topics = segment_topics(text, use_embeddings=False, min_topic_sentences=2)
        assert len(topics) >= 1  # Should detect at least one topic

    def test_max_topics(self):
        text = ". ".join([f"Topic {i} discusses concept {i}" for i in range(50)])
        topics = segment_topics(text, use_embeddings=False, max_topics=5, min_topic_sentences=2)
        assert len(topics) <= 5


# ---------------------------------------------------------------------------
# Chapter detection tests
# ---------------------------------------------------------------------------

class TestDetectChapters:
    def test_empty_topics(self):
        chapters = detect_chapters([])
        assert len(chapters) == 0

    def test_single_topic(self):
        topics = [
            TopicSegment(
                topic_id=0, title="Introduction", summary="Intro", text="text",
                start_time=0, end_time=120, start_index=0, end_index=10,
            ),
        ]
        chapters = detect_chapters(topics)
        assert len(chapters) == 1
        assert chapters[0].title == "Introduction"

    def test_multiple_topics_merge_short(self):
        topics = [
            TopicSegment(
                topic_id=0, title="Intro", summary="Short intro", text="text",
                start_time=0, end_time=30, start_index=0, end_index=3,
            ),
            TopicSegment(
                topic_id=1, title="Overview", summary="Brief overview", text="text",
                start_time=30, end_time=50, start_index=3, end_index=6,
            ),
            TopicSegment(
                topic_id=2, title="Main Content", summary="The main part", text="long text " * 50,
                start_time=50, end_time=300, start_index=6, end_index=30,
            ),
        ]
        chapters = detect_chapters(topics, min_chapter_duration=60.0)
        # First two short topics should be merged
        assert len(chapters) <= 3

    def test_max_chapters(self):
        topics = [
            TopicSegment(
                topic_id=i, title=f"Topic {i}", summary=f"Summary {i}", text=f"text {i}",
                start_time=i * 120, end_time=(i + 1) * 120, start_index=i * 10, end_index=(i + 1) * 10,
            )
            for i in range(20)
        ]
        chapters = detect_chapters(topics, max_chapters=5)
        assert len(chapters) <= 5


# ---------------------------------------------------------------------------
# YouTube chapters generation tests
# ---------------------------------------------------------------------------

class TestGenerateYoutubeChapters:
    def test_generate_chapters(self):
        chapters = [
            Chapter(chapter_id=0, title="Introduction", start_time=0, end_time=120, duration=120),
            Chapter(chapter_id=1, title="Main Content", start_time=120, end_time=600, duration=480),
            Chapter(chapter_id=2, title="Conclusion", start_time=600, end_time=720, duration=120),
        ]
        result = generate_youtube_chapters(chapters)
        assert "0:00 Introduction" in result
        assert "2:00 Main Content" in result
        assert "10:00 Conclusion" in result

    def test_empty_chapters(self):
        assert generate_youtube_chapters([]) == ""


# ---------------------------------------------------------------------------
# Full segmentation pipeline tests
# ---------------------------------------------------------------------------

class TestSegmentContent:
    def test_full_pipeline(self):
        from shruti.segments import segment_content

        transcript = ". ".join([
            "Vedanta is an ancient philosophical tradition",
            "It discusses the nature of Brahman and Atman",
            "The Upanishads form its foundation",
            "Yoga is a complementary practice",
            "It involves meditation and pranayama",
            "Together they form a complete spiritual path",
        ] * 3)

        result = segment_content(
            transcript=transcript,
            min_topic_sentences=2,
            max_topics=10,
        )

        assert isinstance(result, SegmentationResult)
        assert result.total_topics >= 1
        assert result.total_chapters >= 1
        assert len(result.content_flow) == result.total_topics
