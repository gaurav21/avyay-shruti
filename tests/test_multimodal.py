"""
Tests for ŚRUTI V2.0 multi-modal analysis orchestrator.
"""

import pytest
from unittest.mock import patch, MagicMock

from shruti.multimodal import (
    MultiModalResult,
    _merge_all_modalities,
    _summarize_emotions,
)
from shruti.visual import VisualAnalysis, SlideText, DiagramInfo
from shruti.speakers import (
    SpeakerAnalysis, SpeakerProfile, SpeakerSegment, EmotionSegment,
)
from shruti.segments import SegmentationResult, TopicSegment, Chapter
from shruti.knowledge_graph import KnowledgeGraph, Entity, Relationship


# ---------------------------------------------------------------------------
# MultiModalResult tests
# ---------------------------------------------------------------------------

class TestMultiModalResult:
    def test_create_minimal(self):
        result = MultiModalResult(
            video_id="test_video",
            title="Test Video",
            transcript="Hello world",
            language="en",
        )
        assert result.video_id == "test_video"
        assert result.visual_analysis is None
        assert result.speaker_analysis is None
        assert result.segmentation is None
        assert result.knowledge_graph is None
        assert result.errors == []
        assert result.pipeline_stages == {}

    def test_pipeline_stages(self):
        result = MultiModalResult(
            video_id="v1",
            title="Test",
            transcript="text",
            language="en",
            pipeline_stages={
                "visual": True,
                "speakers": False,
                "segmentation": True,
                "knowledge_graph": True,
                "cross_references": False,
            },
        )
        assert sum(1 for v in result.pipeline_stages.values() if v) == 3


# ---------------------------------------------------------------------------
# Emotion summary tests
# ---------------------------------------------------------------------------

class TestSummarizeEmotions:
    def test_summarize_mixed_emotions(self):
        emotions = [
            EmotionSegment(start_time=0, end_time=5, text="", primary_emotion="joy",
                          sentiment="positive", sentiment_score=0.8),
            EmotionSegment(start_time=5, end_time=10, text="", primary_emotion="devotion",
                          sentiment="positive", sentiment_score=0.7),
            EmotionSegment(start_time=10, end_time=15, text="", primary_emotion="sadness",
                          sentiment="negative", sentiment_score=-0.5),
        ]
        summary = _summarize_emotions(emotions)
        assert summary["total_segments_analyzed"] == 3
        assert summary["dominant_emotion"] in ("joy", "devotion")  # One has higher count
        assert "overall_sentiment" in summary

    def test_empty_emotions(self):
        assert _summarize_emotions([]) == {}

    def test_all_positive(self):
        emotions = [
            EmotionSegment(start_time=0, end_time=5, text="", primary_emotion="joy",
                          sentiment="positive", sentiment_score=0.9),
            EmotionSegment(start_time=5, end_time=10, text="", primary_emotion="peace",
                          sentiment="positive", sentiment_score=0.8),
        ]
        summary = _summarize_emotions(emotions)
        assert summary["overall_sentiment"] == "positive"
        assert summary["avg_sentiment_score"] > 0


# ---------------------------------------------------------------------------
# Knowledge enrichment tests
# ---------------------------------------------------------------------------

class TestMergeModalities:
    def test_merge_empty_result(self):
        result = MultiModalResult(
            video_id="v1",
            title="Test",
            transcript="text",
            language="en",
            knowledge={"summary": "Test summary"},
        )
        enriched = _merge_all_modalities(result)
        assert enriched["summary"] == "Test summary"
        assert enriched["_pipeline"]["version"] == "2.0"

    def test_merge_with_visual(self):
        result = MultiModalResult(
            video_id="v1",
            title="Test",
            transcript="text",
            language="en",
            visual_analysis=VisualAnalysis(
                frames_extracted=50,
                slide_texts=[
                    SlideText(timestamp=10.0, text="Slide one", confidence=0.9),
                ],
                diagrams=[
                    DiagramInfo(timestamp=20.0, diagram_type="chart", description="Bar chart"),
                ],
                unique_slides=1,
                scene_changes=[15.0, 45.0],
                visual_summary="Found slides and charts",
            ),
        )
        enriched = _merge_all_modalities(result)
        assert "visual" in enriched
        assert enriched["visual"]["frames_analyzed"] == 50
        assert len(enriched["visual"]["slide_texts"]) == 1
        assert len(enriched["visual"]["diagrams"]) == 1

    def test_merge_with_speakers(self):
        result = MultiModalResult(
            video_id="v1",
            title="Test",
            transcript="text",
            language="en",
            speaker_analysis=SpeakerAnalysis(
                speakers=[
                    SpeakerProfile(speaker_id="S0", total_duration=100,
                                 segment_count=5, speaking_ratio=0.7, avg_segment_duration=20),
                ],
                segments=[],
                emotions=[
                    EmotionSegment(start_time=0, end_time=5, text="joy",
                                  primary_emotion="joy", sentiment="positive", sentiment_score=0.9),
                ],
                total_speakers=1,
                total_duration=100,
                dominant_speaker="S0",
            ),
        )
        enriched = _merge_all_modalities(result)
        assert "speakers" in enriched
        assert enriched["speakers"]["total_speakers"] == 1
        assert "emotions" in enriched["speakers"]

    def test_merge_with_segmentation(self):
        result = MultiModalResult(
            video_id="v1",
            title="Test",
            transcript="text",
            language="en",
            segmentation=SegmentationResult(
                topics=[
                    TopicSegment(topic_id=0, title="Intro", summary="Introduction",
                               start_time=0, end_time=60, start_index=0, end_index=5,
                               text="intro text", keywords=["intro", "begin"]),
                ],
                chapters=[
                    Chapter(chapter_id=0, title="Intro", start_time=0, end_time=60, duration=60),
                ],
                total_topics=1,
                total_chapters=1,
                avg_topic_duration=60.0,
                content_flow=["Intro"],
            ),
        )
        enriched = _merge_all_modalities(result)
        assert "chapters" in enriched
        assert "topics" in enriched
        assert enriched["content_flow"] == ["Intro"]

    def test_merge_with_knowledge_graph(self):
        result = MultiModalResult(
            video_id="v1",
            title="Test",
            transcript="text",
            language="en",
            knowledge_graph=KnowledgeGraph(
                entities={
                    "e1": Entity(entity_id="e1", name="Brahman", entity_type="concept", frequency=5),
                    "e2": Entity(entity_id="e2", name="Atman", entity_type="concept", frequency=3),
                },
                relationships=[
                    Relationship(source_id="e1", target_id="e2", relation_type="related_to"),
                ],
                cross_references=[],
                total_entities=2,
                total_relationships=1,
                graph_metadata={"entity_types": {"concept": 2}},
            ),
        )
        enriched = _merge_all_modalities(result)
        assert "knowledge_graph" in enriched
        assert enriched["knowledge_graph"]["total_entities"] == 2

    def test_merge_all_modalities(self):
        """Test merging all modalities together."""
        result = MultiModalResult(
            video_id="v1",
            title="Complete Test",
            transcript="Complete transcript",
            language="en",
            knowledge={"summary": "Full summary"},
            visual_analysis=VisualAnalysis(
                frames_extracted=10, slide_texts=[], diagrams=[],
                unique_slides=0, scene_changes=[], visual_summary="",
            ),
            speaker_analysis=SpeakerAnalysis(
                speakers=[], segments=[], emotions=[],
                total_speakers=0, total_duration=0,
            ),
            segmentation=SegmentationResult(
                topics=[], chapters=[], total_topics=0, total_chapters=0,
                avg_topic_duration=0, content_flow=[],
            ),
            knowledge_graph=KnowledgeGraph(
                entities={}, relationships=[], cross_references=[],
                total_entities=0, total_relationships=0,
            ),
            pipeline_stages={
                "visual": True, "speakers": True,
                "segmentation": True, "knowledge_graph": True,
            },
        )
        enriched = _merge_all_modalities(result)

        # All sections should be present
        assert "visual" in enriched
        assert "speakers" in enriched
        assert "chapters" in enriched
        assert "knowledge_graph" in enriched
        assert "_pipeline" in enriched
        assert enriched["_pipeline"]["version"] == "2.0"
