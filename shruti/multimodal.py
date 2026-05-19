"""
Multi-modal analysis orchestrator for ŚRUTI V2.0.
Combines audio transcription, visual analysis, speaker identification,
topic segmentation, and knowledge graph into a unified pipeline.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .knowledge_graph import (
    CrossReference, Entity, KnowledgeGraph, Relationship,
    build_knowledge_graph, detect_cross_references,
    extract_entities, extract_relationships,
    load_knowledge_graph, save_knowledge_graph,
)
from .segments import Chapter, SegmentationResult, TopicSegment, segment_content
from .speakers import SpeakerAnalysis, analyze_speakers
from .visual import VisualAnalysis, analyze_video_visuals, cleanup_frames

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MultiModalResult:
    """Complete multi-modal analysis result."""
    video_id: str
    title: str
    transcript: str
    language: str

    # V1 results
    knowledge: Dict[str, Any] = field(default_factory=dict)

    # V2 results
    visual_analysis: Optional[VisualAnalysis] = None
    speaker_analysis: Optional[SpeakerAnalysis] = None
    segmentation: Optional[SegmentationResult] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    cross_references: List[CrossReference] = field(default_factory=list)

    # Merged output
    enriched_knowledge: Dict[str, Any] = field(default_factory=dict)

    # Pipeline metadata
    pipeline_stages: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Multi-modal pipeline
# ---------------------------------------------------------------------------

def analyze_multimodal(
    video_path: str,
    audio_path: str,
    video_id: str,
    title: str,
    transcript: str,
    language: str,
    transcript_segments: Optional[List[Dict]] = None,
    knowledge: Optional[Dict] = None,
    enable_visual: bool = True,
    enable_speakers: bool = True,
    enable_segmentation: bool = True,
    enable_knowledge_graph: bool = True,
    enable_cross_references: bool = True,
    knowledge_graph_path: Optional[str] = None,
    frame_interval: float = 5.0,
    max_frames: int = 200,
    max_speakers: int = 10,
    cleanup_temp: bool = True,
) -> MultiModalResult:
    """
    Run the full multi-modal analysis pipeline.

    Pipeline stages:
    1. Visual analysis (frame extraction, OCR, diagram detection)
    2. Speaker identification (diarization, emotion analysis)
    3. Topic segmentation (boundary detection, chapter generation)
    4. Knowledge graph (entity/relationship extraction, graph building)
    5. Cross-reference detection (linking to existing content)
    6. Knowledge enrichment (merge all modalities)

    Args:
        video_path: Path to video file (for visual analysis)
        audio_path: Path to audio file (for speaker analysis)
        video_id: Video identifier
        title: Video title
        transcript: Full transcript text
        language: Primary language
        transcript_segments: Optional timed segments
        knowledge: V1 extracted knowledge dict
        enable_visual: Run visual analysis
        enable_speakers: Run speaker identification
        enable_segmentation: Run topic segmentation
        enable_knowledge_graph: Build knowledge graph
        enable_cross_references: Detect cross-references
        knowledge_graph_path: Path to persistent knowledge graph
        frame_interval: Seconds between frame captures
        max_frames: Maximum frames to extract
        max_speakers: Maximum expected speakers
        cleanup_temp: Clean up temporary files after processing

    Returns:
        MultiModalResult with all analysis outputs
    """
    import time
    start_time = time.time()

    result = MultiModalResult(
        video_id=video_id,
        title=title,
        transcript=transcript,
        language=language,
        knowledge=knowledge or {},
    )

    frames_dir = None

    # Stage 1: Visual Analysis
    if enable_visual and video_path and os.path.exists(video_path):
        try:
            frames_dir = tempfile.mkdtemp(prefix="shruti_frames_")
            result.visual_analysis = analyze_video_visuals(
                video_path=video_path,
                frame_interval=frame_interval,
                max_frames=max_frames,
                output_dir=frames_dir,
            )
            result.pipeline_stages["visual"] = True
            logger.info(f"Visual analysis complete: {result.visual_analysis.frames_extracted} frames")
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            result.errors.append(f"Visual analysis failed: {e}")
            result.pipeline_stages["visual"] = False
    else:
        result.pipeline_stages["visual"] = False
        if enable_visual:
            result.errors.append("Visual analysis skipped: video file not available")

    # Stage 2: Speaker Identification
    if enable_speakers and audio_path and os.path.exists(audio_path):
        try:
            result.speaker_analysis = analyze_speakers(
                audio_path=audio_path,
                transcript_segments=transcript_segments,
                max_speakers=max_speakers,
            )
            result.pipeline_stages["speakers"] = True
            logger.info(f"Speaker analysis complete: {result.speaker_analysis.total_speakers} speakers")
        except Exception as e:
            logger.error(f"Speaker analysis failed: {e}")
            result.errors.append(f"Speaker analysis failed: {e}")
            result.pipeline_stages["speakers"] = False
    else:
        result.pipeline_stages["speakers"] = False
        if enable_speakers:
            result.errors.append("Speaker analysis skipped: audio file not available")

    # Stage 3: Topic Segmentation
    if enable_segmentation and transcript:
        try:
            result.segmentation = segment_content(
                transcript=transcript,
                segments=transcript_segments,
            )
            result.pipeline_stages["segmentation"] = True
            logger.info(
                f"Segmentation complete: {result.segmentation.total_topics} topics, "
                f"{result.segmentation.total_chapters} chapters"
            )
        except Exception as e:
            logger.error(f"Topic segmentation failed: {e}")
            result.errors.append(f"Topic segmentation failed: {e}")
            result.pipeline_stages["segmentation"] = False
    else:
        result.pipeline_stages["segmentation"] = False

    # Stage 4: Knowledge Graph
    entities: List[Entity] = []
    relationships: List[Relationship] = []

    if enable_knowledge_graph and transcript:
        try:
            # Extract entities
            entities = extract_entities(
                text=transcript,
                video_id=video_id,
                title=title,
                knowledge=knowledge,
                language=language,
            )

            # Extract relationships
            relationships = extract_relationships(
                entities=entities,
                text=transcript,
                knowledge=knowledge,
                video_id=video_id,
            )

            # Load existing graph or create new
            config = get_config()
            graph_path = knowledge_graph_path or os.path.join(
                str(config.get_chroma_persist_path()), "knowledge_graph.json"
            )

            existing_graph = load_knowledge_graph(graph_path)

            # Build/update graph
            result.knowledge_graph = build_knowledge_graph(
                entities=entities,
                relationships=relationships,
                existing_graph=existing_graph,
            )

            # Save updated graph
            save_knowledge_graph(result.knowledge_graph, graph_path)

            result.pipeline_stages["knowledge_graph"] = True
            logger.info(
                f"Knowledge graph updated: {result.knowledge_graph.total_entities} entities, "
                f"{result.knowledge_graph.total_relationships} relationships"
            )
        except Exception as e:
            logger.error(f"Knowledge graph building failed: {e}")
            result.errors.append(f"Knowledge graph building failed: {e}")
            result.pipeline_stages["knowledge_graph"] = False
    else:
        result.pipeline_stages["knowledge_graph"] = False

    # Stage 5: Cross-Reference Detection
    if enable_cross_references and entities and result.knowledge_graph:
        try:
            result.cross_references = detect_cross_references(
                current_entities=entities,
                current_video_id=video_id,
                graph=result.knowledge_graph,
            )

            # Add cross-references to graph
            if result.cross_references:
                result.knowledge_graph = build_knowledge_graph(
                    entities=[],
                    relationships=[],
                    cross_references=result.cross_references,
                    existing_graph=result.knowledge_graph,
                )

                # Re-save with cross-references
                config = get_config()
                graph_path = knowledge_graph_path or os.path.join(
                    str(config.get_chroma_persist_path()), "knowledge_graph.json"
                )
                save_knowledge_graph(result.knowledge_graph, graph_path)

            result.pipeline_stages["cross_references"] = True
            logger.info(f"Cross-references: {len(result.cross_references)} found")
        except Exception as e:
            logger.error(f"Cross-reference detection failed: {e}")
            result.errors.append(f"Cross-reference detection failed: {e}")
            result.pipeline_stages["cross_references"] = False
    else:
        result.pipeline_stages["cross_references"] = False

    # Stage 6: Enrich knowledge
    result.enriched_knowledge = _merge_all_modalities(result)

    # Cleanup
    if cleanup_temp and frames_dir:
        cleanup_frames(frames_dir)

    result.processing_time_seconds = time.time() - start_time
    logger.info(
        f"Multi-modal analysis complete in {result.processing_time_seconds:.1f}s "
        f"({sum(1 for v in result.pipeline_stages.values() if v)}/{len(result.pipeline_stages)} stages succeeded)"
    )

    return result


# ---------------------------------------------------------------------------
# Knowledge enrichment
# ---------------------------------------------------------------------------

def _merge_all_modalities(result: MultiModalResult) -> Dict[str, Any]:
    """
    Merge all analysis modalities into enriched knowledge output.
    Combines V1 extraction with V2 multi-modal data.
    """
    enriched = dict(result.knowledge)

    # Add visual insights
    if result.visual_analysis:
        va = result.visual_analysis
        enriched["visual"] = {
            "frames_analyzed": va.frames_extracted,
            "unique_slides": va.unique_slides,
            "slide_texts": [
                {
                    "timestamp": st.timestamp,
                    "text": st.text[:500],
                    "confidence": st.confidence,
                    "language": st.language,
                }
                for st in va.slide_texts[:20]
            ],
            "diagrams": [
                {
                    "timestamp": d.timestamp,
                    "type": d.diagram_type,
                    "description": d.description,
                }
                for d in va.diagrams[:20]
            ],
            "scene_changes": len(va.scene_changes),
            "visual_summary": va.visual_summary,
        }

    # Add speaker insights
    if result.speaker_analysis:
        sa = result.speaker_analysis
        enriched["speakers"] = {
            "total_speakers": sa.total_speakers,
            "dominant_speaker": sa.dominant_speaker,
            "profiles": [
                {
                    "speaker_id": sp.speaker_id,
                    "label": sp.speaker_label,
                    "speaking_duration": sp.total_duration,
                    "speaking_ratio": sp.speaking_ratio,
                    "segment_count": sp.segment_count,
                }
                for sp in sa.speakers
            ],
            "emotions": _summarize_emotions(sa.emotions) if sa.emotions else {},
        }

    # Add segmentation insights
    if result.segmentation:
        seg = result.segmentation
        enriched["chapters"] = [
            {
                "chapter_id": ch.chapter_id,
                "title": ch.title,
                "start_time": ch.start_time,
                "end_time": ch.end_time,
                "duration": ch.duration,
                "summary": ch.summary,
            }
            for ch in seg.chapters
        ]
        enriched["topics"] = [
            {
                "topic_id": t.topic_id,
                "title": t.title,
                "keywords": t.keywords[:5],
                "start_time": t.start_time,
                "end_time": t.end_time,
            }
            for t in seg.topics
        ]
        enriched["content_flow"] = seg.content_flow

    # Add knowledge graph insights
    if result.knowledge_graph:
        kg = result.knowledge_graph
        enriched["knowledge_graph"] = {
            "total_entities": kg.total_entities,
            "total_relationships": kg.total_relationships,
            "entity_types": kg.graph_metadata.get("entity_types", {}),
            "key_entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "frequency": e.frequency,
                }
                for e in sorted(
                    kg.entities.values(),
                    key=lambda x: x.frequency,
                    reverse=True,
                )[:15]
            ],
        }

    # Add cross-references
    if result.cross_references:
        enriched["cross_references"] = [
            {
                "target_video": cr.target_video,
                "shared_entities": cr.shared_entities[:10],
                "similarity": cr.similarity_score,
                "type": cr.ref_type,
            }
            for cr in result.cross_references[:10]
        ]

    # Pipeline metadata
    enriched["_pipeline"] = {
        "version": "2.0",
        "stages": result.pipeline_stages,
        "errors": result.errors,
        "processing_time": result.processing_time_seconds,
    }

    return enriched


def _summarize_emotions(emotions) -> Dict[str, Any]:
    """Summarize emotion analysis results."""
    if not emotions:
        return {}

    emotion_counts: Dict[str, int] = {}
    sentiment_scores: List[float] = []

    for e in emotions:
        emotion_counts[e.primary_emotion] = emotion_counts.get(e.primary_emotion, 0) + 1
        sentiment_scores.append(e.sentiment_score)

    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
    avg_sentiment = sum(sentiment_scores) / max(len(sentiment_scores), 1)

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_counts,
        "avg_sentiment_score": avg_sentiment,
        "overall_sentiment": "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral",
        "total_segments_analyzed": len(emotions),
    }
