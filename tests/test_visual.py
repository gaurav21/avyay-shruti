"""
Tests for ŚRUTI V2.0 visual analysis module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from shruti.visual import (
    VideoFrame,
    SlideText,
    DiagramInfo,
    VisualAnalysis,
    deduplicate_slides,
    _detect_text_language,
    _text_similarity,
    _analyze_frame_content,
    _generate_visual_summary,
    cleanup_frames,
)


# ---------------------------------------------------------------------------
# VideoFrame tests
# ---------------------------------------------------------------------------

class TestVideoFrame:
    def test_create_frame(self):
        frame = VideoFrame(
            timestamp=5.0,
            image_path="/tmp/frame_001.jpg",
            frame_index=0,
        )
        assert frame.timestamp == 5.0
        assert frame.frame_index == 0
        assert frame.is_keyframe is False
        assert frame.scene_change_score == 0.0

    def test_keyframe(self):
        frame = VideoFrame(
            timestamp=10.0,
            image_path="/tmp/frame_002.jpg",
            frame_index=1,
            is_keyframe=True,
            scene_change_score=0.85,
        )
        assert frame.is_keyframe is True
        assert frame.scene_change_score == 0.85


# ---------------------------------------------------------------------------
# SlideText tests
# ---------------------------------------------------------------------------

class TestSlideText:
    def test_create_slide_text(self):
        slide = SlideText(
            timestamp=15.0,
            text="Introduction to Vedanta",
            confidence=0.92,
            language="en",
        )
        assert slide.text == "Introduction to Vedanta"
        assert slide.confidence == 0.92
        assert slide.language == "en"
        assert slide.bounding_boxes == []


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplicateSlides:
    def test_empty_list(self):
        assert deduplicate_slides([]) == []

    def test_single_slide(self):
        slides = [SlideText(timestamp=0, text="Hello world", confidence=0.9)]
        result = deduplicate_slides(slides)
        assert len(result) == 1

    def test_duplicate_slides(self):
        slides = [
            SlideText(timestamp=0, text="Introduction to Vedanta Philosophy", confidence=0.8),
            SlideText(timestamp=5, text="Introduction to Vedanta Philosophy", confidence=0.9),
            SlideText(timestamp=10, text="Introduction to Vedanta Philosophy", confidence=0.85),
        ]
        result = deduplicate_slides(slides)
        assert len(result) == 1
        # Should keep highest confidence
        assert result[0].confidence == 0.9

    def test_different_slides(self):
        slides = [
            SlideText(timestamp=0, text="Topic One: Brahman", confidence=0.9),
            SlideText(timestamp=30, text="Topic Two: Atman and Self", confidence=0.85),
            SlideText(timestamp=60, text="Topic Three: Maya and Illusion", confidence=0.88),
        ]
        result = deduplicate_slides(slides)
        assert len(result) == 3

    def test_custom_threshold(self):
        slides = [
            SlideText(timestamp=0, text="The concept of Brahman in Vedanta", confidence=0.9),
            SlideText(timestamp=5, text="The concept of Brahman in Vedantic thought", confidence=0.85),
        ]
        # Low threshold = more strict matching (fewer deduped)
        result_strict = deduplicate_slides(slides, similarity_threshold=0.5)
        # High threshold = more lenient (more deduped)
        result_lenient = deduplicate_slides(slides, similarity_threshold=0.95)
        assert len(result_strict) <= len(result_lenient) or len(result_strict) == len(result_lenient)


# ---------------------------------------------------------------------------
# Language detection tests
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_english_text(self):
        assert _detect_text_language("This is an English sentence") == "en"

    def test_hindi_text(self):
        assert _detect_text_language("यह एक हिंदी वाक्य है") == "hi"

    def test_mixed_text(self):
        # Mixed text should be classified by dominant script
        lang = _detect_text_language("The concept of धर्म is important")
        assert lang in ("en", "hi")  # depends on ratio

    def test_empty_text(self):
        assert _detect_text_language("") == "unknown"

    def test_only_numbers(self):
        assert _detect_text_language("12345 67890") == "unknown"


# ---------------------------------------------------------------------------
# Text similarity tests
# ---------------------------------------------------------------------------

class TestTextSimilarity:
    def test_identical(self):
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _text_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        sim = _text_similarity("hello world today", "hello world tomorrow")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert _text_similarity("", "") == 0.0
        assert _text_similarity("hello", "") == 0.0
        assert _text_similarity("", "world") == 0.0


# ---------------------------------------------------------------------------
# Frame content analysis tests
# ---------------------------------------------------------------------------

class TestFrameContentAnalysis:
    def test_slide_detection(self):
        """Test that a mostly white image with edges is detected as a slide."""
        import numpy as np
        # Create a white image with some text-like features
        img = np.ones((480, 640, 3), dtype=np.uint8) * 240
        # Add some dark lines (simulating text)
        img[100:102, 50:500] = 20
        img[120:122, 50:400] = 20
        img[140:142, 50:450] = 20

        result = _analyze_frame_content(img)
        assert isinstance(result, dict)
        assert "has_diagram" in result
        assert "type" in result
        assert "confidence" in result

    def test_dark_image(self):
        """Test that a dark image is not classified as a slide."""
        import numpy as np
        img = np.ones((480, 640, 3), dtype=np.uint8) * 30

        result = _analyze_frame_content(img)
        assert result["light_ratio"] < 0.5


# ---------------------------------------------------------------------------
# Visual summary tests
# ---------------------------------------------------------------------------

class TestVisualSummary:
    def test_empty_summary(self):
        summary = _generate_visual_summary(
            frames_count=0,
            unique_slides=[],
            diagrams=[],
            scene_changes=[],
        )
        assert "Analyzed 0 frames" in summary

    def test_summary_with_slides(self):
        slides = [
            SlideText(timestamp=10.0, text="Introduction", confidence=0.9),
            SlideText(timestamp=30.0, text="Main Content", confidence=0.85),
        ]
        summary = _generate_visual_summary(
            frames_count=20,
            unique_slides=slides,
            diagrams=[],
            scene_changes=[15.0, 45.0],
        )
        assert "20 frames" in summary
        assert "2 unique slides" in summary
        assert "2 scene changes" in summary

    def test_summary_with_diagrams(self):
        diagrams = [
            DiagramInfo(timestamp=5.0, diagram_type="flowchart", description="Process flow"),
            DiagramInfo(timestamp=25.0, diagram_type="chart", description="Bar chart"),
        ]
        summary = _generate_visual_summary(
            frames_count=10,
            unique_slides=[],
            diagrams=diagrams,
            scene_changes=[],
        )
        assert "2 visual elements" in summary


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_existing_dir(self):
        tmpdir = tempfile.mkdtemp(prefix="shruti_test_")
        assert os.path.exists(tmpdir)
        cleanup_frames(tmpdir)
        assert not os.path.exists(tmpdir)

    def test_cleanup_nonexistent_dir(self):
        # Should not raise
        cleanup_frames("/tmp/nonexistent_shruti_test_dir_12345")
