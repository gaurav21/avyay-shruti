"""
Tests for ŚRUTI V3.0 Advanced Video Summarization.
Tests chapter detection, insights extraction, sentiment analysis,
export formats, and the full summarization pipeline.
"""

import json
import os
import tempfile

import pytest

from shruti.insights import (
    ActionItem,
    Insight,
    InsightType,
    InsightsResult,
    SentimentLabel,
    SentimentSegment,
    extract_insights,
)
from shruti.summarize import (
    ChapterDetectionConfig,
    EnhancedChapter,
    SummarizationConfig,
    SummarizationResult,
    VideoSummary,
    compute_summarization_quality,
    detect_chapters_enhanced,
    generate_executive_summary,
    summarize_video,
)
from shruti.exports import (
    export_chapter_timestamps,
    export_summary_json,
    export_summary_markdown,
    save_insights_to_db,
    search_insights,
)


# ---------------------------------------------------------------------------
# Test fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """
Welcome to this comprehensive tutorial on building modern web applications.
Today we'll cover everything from architecture to deployment.
This is going to be an exciting journey through modern web development.

First, let's talk about the architecture. The most important thing to remember 
is that a clean architecture separates concerns properly. You should always 
design your system with scalability in mind. According to research, 78% of 
successful applications follow a layered architecture pattern.

Now let's move on to the frontend. React has become the dominant framework 
for building user interfaces. The key takeaway here is that component-based 
architecture makes code more maintainable and testable.

Make sure to set up proper state management from the start. Don't forget 
to implement proper error handling in your components. This is crucial for 
a great user experience.

Next, let's discuss the backend. Node.js with Express provides an excellent 
foundation for building RESTful APIs. The performance is remarkable and the 
ecosystem is incredible. "The best code is code that doesn't need to exist" 
is a philosophy we should all follow.

What if we could build our entire stack with just JavaScript? That's the 
promise of full-stack JavaScript development. Have you ever thought about 
how powerful that could be?

Security is essential when building web applications. Always ensure you 
validate all user inputs. Never store passwords in plain text. Implement 
CORS properly to prevent cross-origin attacks. This is not optional.

Finally, let's talk about deployment. In conclusion, building modern web 
applications requires understanding architecture, frontend, backend, and 
security. To sum up, always prioritize clean code, security, and performance.

The bottom line is that modern web development is complex but rewarding. 
Remember to start small, iterate quickly, and always keep learning.
"""

SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 8.0, "text": "Welcome to this comprehensive tutorial on building modern web applications."},
    {"start": 8.0, "end": 16.0, "text": "Today we'll cover everything from architecture to deployment."},
    {"start": 16.0, "end": 24.0, "text": "This is going to be an exciting journey through modern web development."},
    {"start": 24.0, "end": 40.0, "text": "First, let's talk about the architecture. The most important thing to remember is that a clean architecture separates concerns properly."},
    {"start": 40.0, "end": 56.0, "text": "You should always design your system with scalability in mind. According to research, 78% of successful applications follow a layered architecture pattern."},
    {"start": 56.0, "end": 72.0, "text": "Now let's move on to the frontend. React has become the dominant framework for building user interfaces."},
    {"start": 72.0, "end": 88.0, "text": "The key takeaway here is that component-based architecture makes code more maintainable and testable."},
    {"start": 88.0, "end": 104.0, "text": "Make sure to set up proper state management from the start."},
    {"start": 104.0, "end": 120.0, "text": "Don't forget to implement proper error handling in your components. This is crucial for a great user experience."},
    {"start": 120.0, "end": 144.0, "text": "Next, let's discuss the backend. Node.js with Express provides an excellent foundation for building RESTful APIs."},
    {"start": 144.0, "end": 168.0, "text": 'The performance is remarkable and the ecosystem is incredible. "The best code is code that doesn\'t need to exist" is a philosophy we should all follow.'},
    {"start": 168.0, "end": 192.0, "text": "What if we could build our entire stack with just JavaScript? That's the promise of full-stack JavaScript development."},
    {"start": 192.0, "end": 210.0, "text": "Have you ever thought about how powerful that could be?"},
    {"start": 210.0, "end": 240.0, "text": "Security is essential when building web applications. Always ensure you validate all user inputs."},
    {"start": 240.0, "end": 264.0, "text": "Never store passwords in plain text. Implement CORS properly to prevent cross-origin attacks. This is not optional."},
    {"start": 264.0, "end": 300.0, "text": "Finally, let's talk about deployment. In conclusion, building modern web applications requires understanding architecture, frontend, backend, and security."},
    {"start": 300.0, "end": 330.0, "text": "To sum up, always prioritize clean code, security, and performance."},
    {"start": 330.0, "end": 360.0, "text": "The bottom line is that modern web development is complex but rewarding. Remember to start small, iterate quickly, and always keep learning."},
]


# ---------------------------------------------------------------------------
# Tests: Insights extraction
# ---------------------------------------------------------------------------

class TestInsightsExtraction:
    def test_extract_insights_basic(self):
        """Test basic insights extraction."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
            title="Web Development Tutorial",
        )
        assert isinstance(result, InsightsResult)
        assert result.total_insights > 0
        assert len(result.insights) > 0

    def test_extract_insights_with_segments(self):
        """Test insights extraction with timed segments."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            segments=SAMPLE_SEGMENTS,
            video_id="test123",
            title="Web Development Tutorial",
        )
        assert result.total_insights > 0
        # Check that timestamps are from segments
        for ins in result.insights:
            assert ins.start_time >= 0

    def test_insight_types_detected(self):
        """Test that multiple insight types are detected."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        types_found = set(ins.insight_type for ins in result.insights)
        # Should find at least 3 different types
        assert len(types_found) >= 3, f"Only found types: {types_found}"

    def test_action_items_detected(self):
        """Test that action items are extracted."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        # The transcript contains several imperative instructions
        assert len(result.action_items) > 0

    def test_questions_detected(self):
        """Test that questions are detected."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        questions = [i for i in result.insights if i.insight_type == InsightType.QUESTION]
        # "What if we could..." and "Have you ever..." should be detected
        assert len(questions) > 0

    def test_statistics_detected(self):
        """Test that statistics/data points are detected."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        stats = [i for i in result.insights if i.insight_type == InsightType.STATISTIC]
        # "78% of successful applications" should be detected
        assert len(stats) > 0

    def test_conclusions_detected(self):
        """Test that conclusions are detected."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        conclusions = [i for i in result.insights if i.insight_type == InsightType.CONCLUSION]
        # "In conclusion", "To sum up", "The bottom line" should be detected
        assert len(conclusions) > 0

    def test_emphasis_detected(self):
        """Test that emphasis patterns are detected."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        emphasis = [i for i in result.insights if i.insight_type == InsightType.EMPHASIS]
        # "The most important thing", "key takeaway" etc.
        assert len(emphasis) > 0

    def test_sentiment_arc(self):
        """Test sentiment arc computation."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        assert len(result.sentiment_segments) > 0
        for seg in result.sentiment_segments:
            assert isinstance(seg.label, SentimentLabel)
            assert -1.0 <= seg.score <= 1.0

    def test_overall_sentiment(self):
        """Test overall sentiment computation."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        assert isinstance(result.overall_sentiment, SentimentLabel)
        assert -1.0 <= result.avg_sentiment_score <= 1.0

    def test_highlight_reel(self):
        """Test highlight reel generation."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        reel = result.highlight_reel
        assert reel.video_id == "test123"
        assert len(reel.highlights) > 0
        assert reel.summary_stats.get("total_insights", 0) > 0

    def test_quality_score(self):
        """Test insights quality score."""
        result = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
        )
        assert 0.0 <= result.quality_score <= 1.0

    def test_min_importance_filter(self):
        """Test that min_importance filters low-importance insights."""
        result_low = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
            min_importance=0.1,
        )
        result_high = extract_insights(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test123",
            min_importance=0.7,
        )
        assert result_low.total_insights >= result_high.total_insights

    def test_empty_transcript(self):
        """Test with empty transcript."""
        result = extract_insights(transcript="", video_id="empty")
        assert result.total_insights == 0
        assert len(result.insights) == 0


# ---------------------------------------------------------------------------
# Tests: Enhanced chapter detection
# ---------------------------------------------------------------------------

class TestEnhancedChapterDetection:
    def test_detect_chapters_basic(self):
        """Test basic enhanced chapter detection."""
        chapters, seg_result = detect_chapters_enhanced(
            transcript=SAMPLE_TRANSCRIPT,
        )
        assert len(chapters) > 0
        assert all(isinstance(ch, EnhancedChapter) for ch in chapters)

    def test_detect_chapters_with_scene_changes(self):
        """Test chapter detection with scene change signals."""
        scene_changes = [25.0, 60.0, 120.0, 210.0, 264.0]
        chapters, _ = detect_chapters_enhanced(
            transcript=SAMPLE_TRANSCRIPT,
            segments=SAMPLE_SEGMENTS,
            scene_changes=scene_changes,
        )
        assert len(chapters) > 0
        # Chapters should have scene change info
        has_scene_changes = any(ch.scene_change_count > 0 for ch in chapters)
        assert has_scene_changes

    def test_detect_chapters_with_speaker_changes(self):
        """Test chapter detection with speaker change signals."""
        speaker_changes = [50.0, 120.0, 210.0]
        chapters, _ = detect_chapters_enhanced(
            transcript=SAMPLE_TRANSCRIPT,
            segments=SAMPLE_SEGMENTS,
            speaker_changes=speaker_changes,
        )
        assert len(chapters) > 0
        has_speaker_change = any(ch.has_speaker_change for ch in chapters)
        assert has_speaker_change

    def test_chapter_confidence_scoring(self):
        """Test that chapters have confidence scores."""
        chapters, _ = detect_chapters_enhanced(
            transcript=SAMPLE_TRANSCRIPT,
        )
        for ch in chapters:
            assert 0.0 <= ch.confidence <= 1.0

    def test_chapter_buffer_zones(self):
        """Test that chapters have buffer zones."""
        chapters, _ = detect_chapters_enhanced(
            transcript=SAMPLE_TRANSCRIPT,
        )
        for ch in chapters:
            assert ch.buffer_start <= ch.start_time
            assert ch.buffer_end >= ch.end_time

    def test_chapter_config(self):
        """Test custom chapter detection config."""
        config = ChapterDetectionConfig(
            max_chapters=5,
            min_chapter_duration=30.0,
        )
        chapters, _ = detect_chapters_enhanced(
            transcript=SAMPLE_TRANSCRIPT,
            config=config,
        )
        assert len(chapters) <= 5


# ---------------------------------------------------------------------------
# Tests: Full summarization pipeline
# ---------------------------------------------------------------------------

class TestSummarization:
    def test_summarize_video_basic(self):
        """Test complete summarization pipeline."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-001",
            title="Web Development Tutorial",
        )
        assert isinstance(result, SummarizationResult)
        assert isinstance(result.summary, VideoSummary)
        assert result.summary.video_id == "test-vid-001"
        assert result.summary.title == "Web Development Tutorial"

    def test_summarize_with_segments(self):
        """Test summarization with timed segments."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-002",
            title="Web Dev Tutorial",
            segments=SAMPLE_SEGMENTS,
            total_duration=360.0,
        )
        assert result.summary.duration == 360.0
        assert len(result.enhanced_chapters) > 0
        assert result.insights.total_insights > 0

    def test_executive_summary_generated(self):
        """Test executive summary generation."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-003",
            title="Web Development Tutorial",
        )
        assert len(result.summary.executive_summary) > 50
        assert len(result.summary.executive_summary) <= 600

    def test_youtube_chapters_generated(self):
        """Test YouTube chapter format generation."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-004",
            title="Tutorial",
        )
        yt_chapters = result.summary.youtube_chapters
        assert isinstance(yt_chapters, str)
        if yt_chapters:
            lines = yt_chapters.strip().split("\n")
            # Should have timestamp format "M:SS Title"
            for line in lines:
                assert " " in line

    def test_quality_report(self):
        """Test quality report generation."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-005",
            title="Tutorial",
        )
        qr = result.quality_report
        assert "overall_quality_score" in qr
        assert "grade" in qr
        assert 0.0 <= qr["overall_quality_score"] <= 1.0
        assert qr["grade"] in ["A", "B", "C", "D", "F"]

    def test_content_flow(self):
        """Test content flow tracking."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-006",
            title="Tutorial",
        )
        assert len(result.summary.content_flow) > 0

    def test_processing_time_tracked(self):
        """Test that processing time is tracked."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="test-vid-007",
            title="Tutorial",
        )
        assert result.processing_time_seconds > 0

    def test_summarize_short_transcript(self):
        """Test summarization with very short transcript."""
        result = summarize_video(
            transcript="This is a short video about AI.",
            video_id="short",
            title="Short Video",
        )
        assert isinstance(result, SummarizationResult)

    def test_summarize_with_config(self):
        """Test summarization with custom config."""
        config = SummarizationConfig(
            max_insights=10,
            min_insight_importance=0.5,
            max_summary_length=200,
        )
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="configured",
            title="Tutorial",
            config=config,
        )
        assert len(result.summary.key_insights) <= 10


# ---------------------------------------------------------------------------
# Tests: Export functionality
# ---------------------------------------------------------------------------

class TestExports:
    def _get_sample_result(self) -> SummarizationResult:
        """Helper to get a summarization result for export tests."""
        return summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="export-test",
            title="Web Development Tutorial",
            segments=SAMPLE_SEGMENTS,
            total_duration=360.0,
        )

    def test_export_markdown(self):
        """Test Markdown export."""
        result = self._get_sample_result()
        md = export_summary_markdown(result)
        assert isinstance(md, str)
        assert "# Web Development Tutorial" in md
        assert "## Executive Summary" in md
        assert "## Chapters" in md
        assert "Quality Score" in md

    def test_export_json(self):
        """Test JSON export."""
        result = self._get_sample_result()
        json_str = export_summary_json(result)
        data = json.loads(json_str)
        assert data["video_id"] == "export-test"
        assert "chapters" in data
        assert "key_insights" in data
        assert "quality_report" in data

    def test_export_youtube_chapters(self):
        """Test YouTube chapter timestamp export."""
        result = self._get_sample_result()
        yt = export_chapter_timestamps(result, format="youtube")
        assert isinstance(yt, str)

    def test_export_srt(self):
        """Test SRT subtitle export."""
        result = self._get_sample_result()
        srt = export_chapter_timestamps(result, format="srt")
        assert "-->" in srt

    def test_export_vtt(self):
        """Test WebVTT export."""
        result = self._get_sample_result()
        vtt = export_chapter_timestamps(result, format="vtt")
        assert "WEBVTT" in vtt
        assert "-->" in vtt

    def test_export_ffmpeg_metadata(self):
        """Test ffmpeg metadata export."""
        result = self._get_sample_result()
        meta = export_chapter_timestamps(result, format="ffmpeg")
        assert ";FFMETADATA1" in meta
        assert "[CHAPTER]" in meta

    def test_invalid_export_format(self):
        """Test that invalid format raises error."""
        result = self._get_sample_result()
        with pytest.raises(ValueError, match="Unsupported format"):
            export_chapter_timestamps(result, format="invalid")

    def test_insights_database_save_and_search(self):
        """Test saving and searching insights database."""
        result = self._get_sample_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_result = save_insights_to_db(result, tmpdir)
            assert save_result["video_id"] == "export-test"
            assert save_result["insights_saved"] > 0

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "insights_index.json"))
            assert os.path.exists(os.path.join(tmpdir, "export-test.json"))

            # Search all
            results = search_insights(tmpdir)
            assert len(results) > 0

            # Search with query
            results = search_insights(tmpdir, query="architecture")
            # May or may not find results depending on content
            assert isinstance(results, list)

            # Search by video_id
            results = search_insights(tmpdir, video_id="export-test")
            assert len(results) > 0

            # Search nonexistent video
            results = search_insights(tmpdir, video_id="nonexistent")
            assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: Quality scoring
# ---------------------------------------------------------------------------

class TestQualityScoring:
    def test_quality_score_range(self):
        """Test that quality score is in valid range."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="quality",
            title="Tutorial",
            total_duration=360.0,
        )
        qr = result.quality_report
        assert 0.0 <= qr["overall_quality_score"] <= 1.0

    def test_quality_improves_with_content(self):
        """Test that richer content gets higher quality."""
        # Short transcript
        result_short = summarize_video(
            transcript="Hello world. This is short.",
            video_id="short",
            title="Short",
            total_duration=5.0,
        )

        # Rich transcript
        result_rich = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="rich",
            title="Rich Tutorial",
            total_duration=360.0,
        )

        # Rich content should generally score higher
        assert result_rich.quality_report["overall_quality_score"] >= result_short.quality_report["overall_quality_score"]

    def test_grade_assignment(self):
        """Test grade assignment based on quality score."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="graded",
            title="Tutorial",
        )
        assert result.quality_report["grade"] in ["A", "B", "C", "D", "F"]


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_transcript(self):
        """Test with empty transcript."""
        result = summarize_video(transcript="", video_id="empty", title="Empty")
        assert isinstance(result, SummarizationResult)

    def test_single_sentence(self):
        """Test with single sentence transcript."""
        result = summarize_video(
            transcript="This is a single sentence about machine learning.",
            video_id="single",
            title="Single",
        )
        assert isinstance(result, SummarizationResult)

    def test_non_english_content(self):
        """Test with Hindi content."""
        hindi_transcript = """
        नमस्ते, आज हम कृत्रिम बुद्धिमत्ता के बारे में बात करेंगे।
        यह बहुत महत्वपूर्ण विषय है। आपको हमेशा नई तकनीक सीखते रहना चाहिए।
        मशीन लर्निंग एक शक्तिशाली उपकरण है। इसका उपयोग कई क्षेत्रों में होता है।
        """
        result = summarize_video(
            transcript=hindi_transcript,
            video_id="hindi",
            title="कृत्रिम बुद्धिमत्ता",
        )
        assert isinstance(result, SummarizationResult)

    def test_max_insights_limit(self):
        """Test max_insights limit is respected."""
        config = SummarizationConfig(max_insights=5)
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            video_id="limited",
            title="Tutorial",
            config=config,
        )
        assert len(result.summary.key_insights) <= 5

    def test_no_video_id_generates_one(self):
        """Test that missing video_id is auto-generated."""
        result = summarize_video(
            transcript=SAMPLE_TRANSCRIPT,
            title="No ID",
        )
        assert result.summary.video_id  # Should be auto-generated
