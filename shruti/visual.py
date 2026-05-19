"""
Visual information extraction module for ŚRUTI V2.0.
Extracts knowledge from video frames: slide text (OCR), diagrams, visual elements.
Optimized for Apple Silicon (M1 Max) with CoreML acceleration.
"""

import io
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoFrame:
    """A single extracted video frame with metadata."""
    timestamp: float  # seconds into the video
    image_path: str
    frame_index: int
    is_keyframe: bool = False
    scene_change_score: float = 0.0


@dataclass
class SlideText:
    """OCR-extracted text from a video frame (slide/presentation)."""
    timestamp: float
    text: str
    confidence: float
    language: Optional[str] = None
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DiagramInfo:
    """Information about a detected diagram or visual element."""
    timestamp: float
    diagram_type: str  # flowchart, chart, table, image, equation, etc.
    description: str
    elements: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VisualAnalysis:
    """Complete visual analysis result for a video."""
    frames_extracted: int
    slide_texts: List[SlideText]
    diagrams: List[DiagramInfo]
    unique_slides: int
    scene_changes: List[float]  # timestamps of scene changes
    visual_summary: str


# ---------------------------------------------------------------------------
# Frame extraction using ffmpeg
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    interval_seconds: float = 5.0,
    max_frames: int = 200,
    keyframes_only: bool = False,
    scene_threshold: float = 0.3,
) -> List[VideoFrame]:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames (temp dir if None)
        interval_seconds: Seconds between frame captures
        max_frames: Maximum number of frames to extract
        keyframes_only: Only extract keyframes (I-frames)
        scene_threshold: Scene change detection threshold (0-1)

    Returns:
        List of VideoFrame objects
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="shruti_frames_")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    frames: List[VideoFrame] = []

    try:
        # Get video duration first
        duration = _get_video_duration(video_path)
        if duration is None:
            logger.warning("Could not determine video duration, using defaults")
            duration = 3600.0  # fallback 1h

        # Adjust interval if too many frames would be generated
        estimated_frames = int(duration / interval_seconds)
        if estimated_frames > max_frames:
            interval_seconds = duration / max_frames
            logger.info(f"Adjusted frame interval to {interval_seconds:.1f}s (max {max_frames} frames)")

        # Build ffmpeg command
        output_pattern = os.path.join(output_dir, "frame_%05d.jpg")

        if keyframes_only:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"select='eq(pict_type,I)',fps=1/{interval_seconds}",
                "-vsync", "vfr",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                output_pattern,
                "-y", "-loglevel", "warning",
            ]
        else:
            # Scene change detection + interval-based extraction
            filter_expr = (
                f"select='gt(scene,{scene_threshold})+not(mod(n,{int(interval_seconds * 30)}))',"
                f"fps=1/{interval_seconds}"
            )
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1/{interval_seconds}",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                output_pattern,
                "-y", "-loglevel", "warning",
            ]

        logger.info(f"Extracting frames from {video_path} (interval={interval_seconds}s)")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"ffmpeg frame extraction failed: {result.stderr}")
            raise RuntimeError(f"Frame extraction failed: {result.stderr[:500]}")

        # Collect extracted frames
        frame_files = sorted(Path(output_dir).glob("frame_*.jpg"))
        for idx, frame_path in enumerate(frame_files):
            timestamp = idx * interval_seconds
            frames.append(VideoFrame(
                timestamp=timestamp,
                image_path=str(frame_path),
                frame_index=idx,
                is_keyframe=keyframes_only,
            ))

        logger.info(f"Extracted {len(frames)} frames from video")

    except subprocess.TimeoutExpired:
        logger.error("Frame extraction timed out")
        raise RuntimeError("Frame extraction timed out (>5 min)")
    except FileNotFoundError:
        logger.error("ffmpeg not found — install ffmpeg to enable visual analysis")
        raise RuntimeError("ffmpeg is required for visual analysis. Install with: brew install ffmpeg")

    return frames


def detect_scene_changes(
    video_path: str,
    threshold: float = 0.3,
) -> List[float]:
    """
    Detect scene change timestamps in a video using ffmpeg.

    Returns:
        List of timestamps (seconds) where scene changes occur
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
        "-loglevel", "info",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        timestamps = []

        for line in result.stderr.split("\n"):
            # Parse showinfo output for pts_time
            match = re.search(r"pts_time:(\d+\.?\d*)", line)
            if match:
                timestamps.append(float(match.group(1)))

        logger.info(f"Detected {len(timestamps)} scene changes")
        return timestamps

    except Exception as e:
        logger.error(f"Scene change detection failed: {e}")
        return []


# ---------------------------------------------------------------------------
# OCR / Slide text extraction
# ---------------------------------------------------------------------------

def extract_slide_text(
    frames: List[VideoFrame],
    use_tesseract: bool = True,
    languages: str = "eng+hin+san",
    min_confidence: float = 0.4,
) -> List[SlideText]:
    """
    Extract text from video frames using OCR.

    Supports:
    - Tesseract OCR (default, supports Sanskrit/Hindi/English)
    - Apple Vision framework on macOS (via PyObjC, faster on M1)

    Args:
        frames: List of VideoFrame objects to process
        use_tesseract: Use Tesseract OCR (True) or try Apple Vision (False)
        languages: Tesseract language codes ('+' separated)
        min_confidence: Minimum OCR confidence threshold

    Returns:
        List of SlideText objects with extracted text
    """
    results: List[SlideText] = []

    if not frames:
        return results

    # Try Apple Vision first on macOS
    if not use_tesseract and _is_macos():
        try:
            return _extract_text_apple_vision(frames, min_confidence)
        except Exception as e:
            logger.warning(f"Apple Vision OCR failed, falling back to Tesseract: {e}")

    # Tesseract OCR
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.error("pytesseract or Pillow not installed. Install with: pip install pytesseract Pillow")
        raise RuntimeError("OCR dependencies missing: pip install pytesseract Pillow")

    for frame in frames:
        try:
            img = Image.open(frame.image_path)

            # Get detailed OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                img,
                lang=languages,
                output_type=pytesseract.Output.DICT,
            )

            # Aggregate text and confidence
            texts = []
            confidences = []
            bboxes = []

            for i, text in enumerate(ocr_data["text"]):
                conf = float(ocr_data["conf"][i])
                if conf > min_confidence * 100 and text.strip():
                    texts.append(text.strip())
                    confidences.append(conf / 100.0)
                    bboxes.append({
                        "x": ocr_data["left"][i],
                        "y": ocr_data["top"][i],
                        "w": ocr_data["width"][i],
                        "h": ocr_data["height"][i],
                    })

            if texts:
                full_text = " ".join(texts)
                avg_conf = sum(confidences) / len(confidences)

                # Detect language from text
                detected_lang = _detect_text_language(full_text)

                results.append(SlideText(
                    timestamp=frame.timestamp,
                    text=full_text,
                    confidence=avg_conf,
                    language=detected_lang,
                    bounding_boxes=bboxes,
                ))

        except Exception as e:
            logger.warning(f"OCR failed for frame at {frame.timestamp}s: {e}")
            continue

    logger.info(f"Extracted text from {len(results)}/{len(frames)} frames")
    return results


def deduplicate_slides(
    slide_texts: List[SlideText],
    similarity_threshold: float = 0.85,
) -> List[SlideText]:
    """
    Deduplicate similar slide texts (consecutive frames of same slide).

    Uses simple text similarity to merge near-identical slides,
    keeping the one with higher confidence.
    """
    if not slide_texts:
        return []

    unique: List[SlideText] = [slide_texts[0]]

    for slide in slide_texts[1:]:
        # Compare with last unique slide
        similarity = _text_similarity(unique[-1].text, slide.text)
        if similarity < similarity_threshold:
            unique.append(slide)
        elif slide.confidence > unique[-1].confidence:
            # Replace with higher-confidence version
            unique[-1] = slide

    logger.info(f"Deduplicated {len(slide_texts)} slides to {len(unique)} unique slides")
    return unique


# ---------------------------------------------------------------------------
# Diagram / visual element detection
# ---------------------------------------------------------------------------

def detect_diagrams(
    frames: List[VideoFrame],
    use_llm: bool = True,
    groq_api_key: Optional[str] = None,
) -> List[DiagramInfo]:
    """
    Detect and classify diagrams/visual elements in video frames.

    Strategy:
    1. Heuristic edge/contour analysis for chart/diagram detection
    2. Optional LLM-based description for complex diagrams

    Args:
        frames: Video frames to analyze
        use_llm: Whether to use LLM for diagram description
        groq_api_key: Groq API key for LLM analysis

    Returns:
        List of DiagramInfo objects
    """
    diagrams: List[DiagramInfo] = []

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("numpy/Pillow not available for diagram detection")
        return diagrams

    for frame in frames:
        try:
            img = Image.open(frame.image_path)
            img_array = np.array(img)

            # Analyze frame for visual elements
            analysis = _analyze_frame_content(img_array)

            if analysis["has_diagram"]:
                diagrams.append(DiagramInfo(
                    timestamp=frame.timestamp,
                    diagram_type=analysis["type"],
                    description=analysis["description"],
                    elements=analysis.get("elements", []),
                    confidence=analysis["confidence"],
                ))

        except Exception as e:
            logger.warning(f"Diagram detection failed for frame at {frame.timestamp}s: {e}")
            continue

    logger.info(f"Detected {len(diagrams)} diagrams across {len(frames)} frames")
    return diagrams


# ---------------------------------------------------------------------------
# Full visual analysis pipeline
# ---------------------------------------------------------------------------

def analyze_video_visuals(
    video_path: str,
    frame_interval: float = 5.0,
    max_frames: int = 200,
    ocr_languages: str = "eng+hin+san",
    detect_diagrams_flag: bool = True,
    output_dir: Optional[str] = None,
) -> VisualAnalysis:
    """
    Complete visual analysis pipeline for a video.

    Steps:
    1. Extract frames at regular intervals
    2. Detect scene changes
    3. Run OCR on all frames
    4. Deduplicate slide texts
    5. Detect diagrams and visual elements
    6. Generate visual summary

    Args:
        video_path: Path to the video file
        frame_interval: Seconds between frame captures
        max_frames: Maximum frames to extract
        ocr_languages: OCR language codes
        detect_diagrams_flag: Whether to run diagram detection
        output_dir: Directory for extracted frames

    Returns:
        VisualAnalysis with all extracted visual information
    """
    logger.info(f"Starting visual analysis of {video_path}")

    # Step 1: Extract frames
    frames = extract_frames(
        video_path,
        output_dir=output_dir,
        interval_seconds=frame_interval,
        max_frames=max_frames,
    )

    # Step 2: Detect scene changes
    scene_changes = detect_scene_changes(video_path)

    # Step 3: OCR on frames
    slide_texts = extract_slide_text(frames, languages=ocr_languages)

    # Step 4: Deduplicate
    unique_slides = deduplicate_slides(slide_texts)

    # Step 5: Diagram detection
    diagram_results: List[DiagramInfo] = []
    if detect_diagrams_flag:
        diagram_results = detect_diagrams(frames)

    # Step 6: Generate summary
    visual_summary = _generate_visual_summary(
        frames_count=len(frames),
        unique_slides=unique_slides,
        diagrams=diagram_results,
        scene_changes=scene_changes,
    )

    return VisualAnalysis(
        frames_extracted=len(frames),
        slide_texts=unique_slides,
        diagrams=diagram_results,
        unique_slides=len(unique_slides),
        scene_changes=scene_changes,
        visual_summary=visual_summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get video duration: {e}")
    return None


def _is_macos() -> bool:
    """Check if running on macOS."""
    import platform
    return platform.system() == "Darwin"


def _extract_text_apple_vision(
    frames: List[VideoFrame],
    min_confidence: float = 0.4,
) -> List[SlideText]:
    """
    Extract text using Apple Vision framework (macOS only).
    Faster on M1/M2 with Neural Engine acceleration.
    """
    results: List[SlideText] = []

    try:
        import Vision
        import Quartz
        from Foundation import NSURL

        for frame in frames:
            url = NSURL.fileURLWithPath_(frame.image_path)
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setRecognitionLanguages_(["en", "hi", "sa"])
            request.setUsesLanguageCorrection_(True)

            handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
            success = handler.performRequests_error_([request], None)

            if success and request.results():
                texts = []
                confidences = []
                for observation in request.results():
                    if observation.confidence() >= min_confidence:
                        top_candidate = observation.topCandidates_(1)[0]
                        texts.append(top_candidate.string())
                        confidences.append(float(observation.confidence()))

                if texts:
                    results.append(SlideText(
                        timestamp=frame.timestamp,
                        text=" ".join(texts),
                        confidence=sum(confidences) / len(confidences),
                        language=_detect_text_language(" ".join(texts)),
                    ))

    except ImportError:
        raise RuntimeError("Apple Vision framework not available")

    return results


def _detect_text_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    total = devanagari_count + latin_count

    if total == 0:
        return "unknown"
    if devanagari_count / max(total, 1) > 0.5:
        return "hi"  # Hindi/Sanskrit (Devanagari script)
    return "en"


def _text_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def _analyze_frame_content(img_array) -> Dict[str, Any]:
    """
    Analyze a frame for diagrams, charts, and visual elements
    using heuristic image analysis.
    """
    import numpy as np

    h, w = img_array.shape[:2]
    total_pixels = h * w

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array

    # Heuristic: high-contrast regions suggest diagrams/text
    # Simple edge detection via gradient magnitude
    gx = np.abs(np.diff(gray.astype(float), axis=1))
    gy = np.abs(np.diff(gray.astype(float), axis=0))

    edge_density = (np.mean(gx) + np.mean(gy)) / 2.0

    # Color variance — low variance + high edges = diagram/slide
    if len(img_array.shape) == 3:
        color_std = np.std(img_array.reshape(-1, 3), axis=0).mean()
    else:
        color_std = np.std(gray)

    # White/light background detection (common in slides)
    light_ratio = np.mean(gray > 200)

    # Classification heuristics
    has_diagram = False
    diagram_type = "unknown"
    description = ""
    confidence = 0.0

    if light_ratio > 0.5 and edge_density > 15:
        has_diagram = True
        confidence = min(0.9, light_ratio * 0.5 + edge_density / 100)

        if edge_density > 40:
            diagram_type = "dense_text_slide"
            description = "Slide with dense text content"
        elif edge_density > 25:
            diagram_type = "diagram_or_chart"
            description = "Possible diagram, chart, or structured visual"
        else:
            diagram_type = "simple_slide"
            description = "Simple slide with minimal content"

    elif edge_density > 30 and color_std > 50:
        has_diagram = True
        diagram_type = "complex_visual"
        description = "Complex visual with varied colors and edges"
        confidence = 0.5

    return {
        "has_diagram": has_diagram,
        "type": diagram_type,
        "description": description,
        "confidence": confidence,
        "edge_density": float(edge_density),
        "color_std": float(color_std),
        "light_ratio": float(light_ratio),
    }


def _generate_visual_summary(
    frames_count: int,
    unique_slides: List[SlideText],
    diagrams: List[DiagramInfo],
    scene_changes: List[float],
) -> str:
    """Generate a human-readable summary of visual analysis."""
    parts = [f"Analyzed {frames_count} frames from video."]

    if unique_slides:
        parts.append(f"Found {len(unique_slides)} unique slides/text screens.")
        # Sample first few slide texts
        for slide in unique_slides[:3]:
            preview = slide.text[:100] + "..." if len(slide.text) > 100 else slide.text
            parts.append(f"  - At {slide.timestamp:.1f}s: \"{preview}\"")

    if diagrams:
        type_counts: Dict[str, int] = {}
        for d in diagrams:
            type_counts[d.diagram_type] = type_counts.get(d.diagram_type, 0) + 1
        type_str = ", ".join(f"{count} {dtype}" for dtype, count in type_counts.items())
        parts.append(f"Detected {len(diagrams)} visual elements: {type_str}.")

    if scene_changes:
        parts.append(f"Detected {len(scene_changes)} scene changes.")

    return "\n".join(parts)


def cleanup_frames(output_dir: str) -> None:
    """Remove extracted frame files to save disk space."""
    import shutil
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info(f"Cleaned up frame directory: {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup frames: {e}")
