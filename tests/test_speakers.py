"""
Tests for ŚRUTI V2.0 speaker identification and emotion analysis module.
"""

import pytest
from unittest.mock import patch, MagicMock

from shruti.speakers import (
    SpeakerSegment,
    SpeakerProfile,
    EmotionSegment,
    SpeakerAnalysis,
    _build_speaker_profiles,
    _align_text_to_segment,
    _analyze_emotions_lexicon,
    _segments_from_transcript,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestSpeakerSegment:
    def test_create_segment(self):
        seg = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=5.0,
            text="Hello everyone",
            confidence=0.8,
        )
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.end_time == 5.0
        assert seg.speaker_label is None

    def test_segment_with_label(self):
        seg = SpeakerSegment(
            speaker_id="SPEAKER_01",
            speaker_label="Guru",
            start_time=10.0,
            end_time=30.0,
        )
        assert seg.speaker_label == "Guru"


class TestSpeakerProfile:
    def test_create_profile(self):
        profile = SpeakerProfile(
            speaker_id="SPEAKER_00",
            total_duration=120.0,
            segment_count=10,
            speaking_ratio=0.6,
            avg_segment_duration=12.0,
        )
        assert profile.total_duration == 120.0
        assert profile.speaking_ratio == 0.6


# ---------------------------------------------------------------------------
# Speaker profile building tests
# ---------------------------------------------------------------------------

class TestBuildSpeakerProfiles:
    def test_single_speaker(self):
        speaker_data = {
            "SPEAKER_00": {"duration": 120.0, "count": 10, "texts": ["Hello", "World"]},
        }
        profiles = _build_speaker_profiles(speaker_data, 200.0)
        assert len(profiles) == 1
        assert profiles[0].speaker_id == "SPEAKER_00"
        assert profiles[0].speaking_ratio == pytest.approx(0.6)

    def test_multiple_speakers(self):
        speaker_data = {
            "SPEAKER_00": {"duration": 120.0, "count": 10, "texts": []},
            "SPEAKER_01": {"duration": 80.0, "count": 5, "texts": []},
        }
        profiles = _build_speaker_profiles(speaker_data, 200.0)
        assert len(profiles) == 2
        # Should be sorted by duration (descending)
        assert profiles[0].speaker_id == "SPEAKER_00"
        assert profiles[1].speaker_id == "SPEAKER_01"

    def test_empty_data(self):
        profiles = _build_speaker_profiles({}, 0.0)
        assert len(profiles) == 0


# ---------------------------------------------------------------------------
# Text alignment tests
# ---------------------------------------------------------------------------

class TestAlignText:
    def test_align_overlapping_segments(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 10.0, "text": "World"},
            {"start": 10.0, "end": 15.0, "text": "Goodbye"},
        ]
        result = _align_text_to_segment(segments, 3.0, 12.0)
        assert "Hello" in result
        assert "World" in result
        assert "Goodbye" in result

    def test_no_overlap(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 20.0, "end": 25.0, "text": "World"},
        ]
        result = _align_text_to_segment(segments, 10.0, 15.0)
        assert result == ""

    def test_empty_segments(self):
        result = _align_text_to_segment([], 0.0, 10.0)
        assert result == ""


# ---------------------------------------------------------------------------
# Transcript-based segments tests
# ---------------------------------------------------------------------------

class TestSegmentsFromTranscript:
    def test_create_from_transcript(self):
        transcript_segs = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 5.0, "end": 10.0, "text": "Second segment"},
        ]
        segments, profiles = _segments_from_transcript(transcript_segs)
        assert len(segments) == 2
        assert len(profiles) == 1
        assert profiles[0].speaker_id == "SPEAKER_00"
        assert segments[0].text == "First segment"

    def test_empty_transcript(self):
        segments, profiles = _segments_from_transcript([])
        assert len(segments) == 0
        assert len(profiles) == 1


# ---------------------------------------------------------------------------
# Emotion analysis (lexicon-based) tests
# ---------------------------------------------------------------------------

class TestLexiconEmotionAnalysis:
    def test_positive_emotion(self):
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                text="This is a wonderful and beautiful teaching about divine love and bliss",
            ),
        ]
        results = _analyze_emotions_lexicon(segments)
        assert len(results) == 1
        assert results[0].sentiment in ("positive", "neutral")

    def test_negative_emotion(self):
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                text="There was great sorrow and suffering and pain in the darkness",
            ),
        ]
        results = _analyze_emotions_lexicon(segments)
        assert len(results) == 1
        assert results[0].primary_emotion in ("sadness", "fear")

    def test_devotional_content(self):
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                text="Through devotion and surrender we worship the divine with prayer and श्रद्धा",
            ),
        ]
        results = _analyze_emotions_lexicon(segments)
        assert len(results) == 1
        assert results[0].primary_emotion in ("devotion", "reverence", "joy")

    def test_neutral_text(self):
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                text="The temperature today measured at thirty two degrees celsius",
            ),
        ]
        results = _analyze_emotions_lexicon(segments)
        assert len(results) == 1
        assert results[0].primary_emotion == "neutral"

    def test_empty_segments(self):
        results = _analyze_emotions_lexicon([])
        assert len(results) == 0

    def test_short_text_skipped(self):
        segments = [
            SpeakerSegment(speaker_id="S", start_time=0, end_time=1, text="Hi"),
        ]
        results = _analyze_emotions_lexicon(segments)
        assert len(results) == 0  # Too short

    def test_sanskrit_emotions(self):
        segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=5.0,
                text="आनन्द और सुख की प्राप्ति भक्ति से होती है meditation peace",
            ),
        ]
        results = _analyze_emotions_lexicon(segments)
        assert len(results) == 1
        # Should detect positive emotions from Sanskrit keywords
        assert results[0].sentiment in ("positive", "neutral")


# ---------------------------------------------------------------------------
# SpeakerAnalysis tests
# ---------------------------------------------------------------------------

class TestSpeakerAnalysis:
    def test_create_analysis(self):
        analysis = SpeakerAnalysis(
            speakers=[],
            segments=[],
            emotions=[],
            total_speakers=0,
            total_duration=0.0,
        )
        assert analysis.dominant_speaker is None

    def test_with_speakers(self):
        analysis = SpeakerAnalysis(
            speakers=[
                SpeakerProfile(speaker_id="S0", total_duration=100, segment_count=5,
                             speaking_ratio=0.7, avg_segment_duration=20),
                SpeakerProfile(speaker_id="S1", total_duration=40, segment_count=3,
                             speaking_ratio=0.3, avg_segment_duration=13.3),
            ],
            segments=[],
            emotions=[],
            total_speakers=2,
            total_duration=140.0,
            dominant_speaker="S0",
        )
        assert analysis.total_speakers == 2
        assert analysis.dominant_speaker == "S0"
