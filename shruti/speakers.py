"""
Speaker identification and emotion analysis module for ŚRUTI V2.0.
Provides speaker diarization (who spoke when) and basic sentiment/emotion analysis.
"""

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpeakerSegment:
    """A segment of speech attributed to a specific speaker."""
    speaker_id: str  # e.g., "SPEAKER_00", "SPEAKER_01"
    speaker_label: Optional[str] = None  # optional human-readable label
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    confidence: float = 0.0


@dataclass
class SpeakerProfile:
    """Profile information for a detected speaker."""
    speaker_id: str
    speaker_label: Optional[str] = None
    total_duration: float = 0.0  # total speaking time in seconds
    segment_count: int = 0
    speaking_ratio: float = 0.0  # ratio of total video duration
    avg_segment_duration: float = 0.0
    sample_texts: List[str] = field(default_factory=list)


@dataclass
class EmotionSegment:
    """Emotion/sentiment analysis for a text segment."""
    start_time: float
    end_time: float
    text: str
    primary_emotion: str  # joy, sadness, anger, fear, surprise, neutral, devotion, reverence
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    sentiment: str = "neutral"  # positive, negative, neutral
    sentiment_score: float = 0.0


@dataclass
class SpeakerAnalysis:
    """Complete speaker analysis result."""
    speakers: List[SpeakerProfile]
    segments: List[SpeakerSegment]
    emotions: List[EmotionSegment]
    total_speakers: int
    total_duration: float
    dominant_speaker: Optional[str] = None


# ---------------------------------------------------------------------------
# Speaker diarization
# ---------------------------------------------------------------------------

def identify_speakers(
    audio_path: str,
    transcript_segments: Optional[List[Dict]] = None,
    max_speakers: int = 10,
    min_segment_duration: float = 0.5,
) -> Tuple[List[SpeakerSegment], List[SpeakerProfile]]:
    """
    Identify different speakers in an audio file.

    Strategy:
    1. Try pyannote.audio for neural speaker diarization (best quality)
    2. Fall back to energy-based segmentation if pyannote unavailable
    3. Align with transcript segments if available

    Args:
        audio_path: Path to the audio file
        transcript_segments: Optional transcript segments with timing
        max_speakers: Maximum number of expected speakers
        min_segment_duration: Minimum segment duration in seconds

    Returns:
        Tuple of (speaker segments, speaker profiles)
    """
    # Try pyannote.audio first
    try:
        return _diarize_with_pyannote(audio_path, transcript_segments, max_speakers)
    except ImportError:
        logger.info("pyannote.audio not available, using energy-based diarization")
    except Exception as e:
        logger.warning(f"pyannote diarization failed: {e}, falling back to energy-based")

    # Fallback: energy-based speaker segmentation
    return _diarize_energy_based(audio_path, transcript_segments, min_segment_duration)


def _diarize_with_pyannote(
    audio_path: str,
    transcript_segments: Optional[List[Dict]],
    max_speakers: int,
) -> Tuple[List[SpeakerSegment], List[SpeakerProfile]]:
    """
    Speaker diarization using pyannote.audio.
    Requires: pip install pyannote.audio
    """
    from pyannote.audio import Pipeline
    import torch

    # Use MPS (Metal) on Apple Silicon if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_TOKEN required for pyannote.audio speaker diarization")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))

    diarization = pipeline(audio_path, max_speakers=max_speakers)

    segments: List[SpeakerSegment] = []
    speaker_data: Dict[str, Dict] = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg = SpeakerSegment(
            speaker_id=speaker,
            start_time=turn.start,
            end_time=turn.end,
            confidence=0.8,  # pyannote doesn't expose per-segment confidence
        )

        # Align with transcript if available
        if transcript_segments:
            seg.text = _align_text_to_segment(
                transcript_segments, turn.start, turn.end
            )

        segments.append(seg)

        # Track speaker stats
        if speaker not in speaker_data:
            speaker_data[speaker] = {"duration": 0.0, "count": 0, "texts": []}
        speaker_data[speaker]["duration"] += (turn.end - turn.start)
        speaker_data[speaker]["count"] += 1
        if seg.text:
            speaker_data[speaker]["texts"].append(seg.text[:200])

    # Build profiles
    total_duration = max(seg.end_time for seg in segments) if segments else 0.0
    profiles = _build_speaker_profiles(speaker_data, total_duration)

    logger.info(f"Diarization complete: {len(profiles)} speakers, {len(segments)} segments")
    return segments, profiles


def _diarize_energy_based(
    audio_path: str,
    transcript_segments: Optional[List[Dict]],
    min_segment_duration: float,
) -> Tuple[List[SpeakerSegment], List[SpeakerProfile]]:
    """
    Simple energy-based speaker segmentation.
    Uses silence detection to identify speaker turns.
    Not as accurate as neural diarization but works without ML dependencies.
    """
    segments: List[SpeakerSegment] = []

    # Use ffmpeg to detect silence boundaries
    silence_timestamps = _detect_silence(audio_path)

    if not silence_timestamps and transcript_segments:
        # Use transcript segments directly as speaker turns
        return _segments_from_transcript(transcript_segments)

    # Create segments between silence boundaries
    audio_duration = _get_audio_duration(audio_path)
    boundaries = [0.0] + silence_timestamps + [audio_duration]

    speaker_counter = 0
    current_speaker = "SPEAKER_00"
    speaker_data: Dict[str, Dict] = {}

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start

        if duration < min_segment_duration:
            continue

        # Simple heuristic: alternate speakers on long silences
        # (This is a rough approximation without ML)
        if i > 0 and (boundaries[i] - boundaries[i - 1]) > 1.5:
            speaker_counter = (speaker_counter + 1) % 2
            current_speaker = f"SPEAKER_{speaker_counter:02d}"

        seg = SpeakerSegment(
            speaker_id=current_speaker,
            start_time=start,
            end_time=end,
            confidence=0.4,  # low confidence for energy-based
        )

        if transcript_segments:
            seg.text = _align_text_to_segment(transcript_segments, start, end)

        segments.append(seg)

        if current_speaker not in speaker_data:
            speaker_data[current_speaker] = {"duration": 0.0, "count": 0, "texts": []}
        speaker_data[current_speaker]["duration"] += duration
        speaker_data[current_speaker]["count"] += 1
        if seg.text:
            speaker_data[current_speaker]["texts"].append(seg.text[:200])

    total_duration = audio_duration
    profiles = _build_speaker_profiles(speaker_data, total_duration)

    logger.info(f"Energy-based diarization: {len(profiles)} speakers, {len(segments)} segments")
    return segments, profiles


def _segments_from_transcript(
    transcript_segments: List[Dict],
) -> Tuple[List[SpeakerSegment], List[SpeakerProfile]]:
    """Create speaker segments directly from transcript segments (single speaker assumed)."""
    segments = []
    for seg_data in transcript_segments:
        segments.append(SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=seg_data.get("start", 0.0),
            end_time=seg_data.get("end", 0.0),
            text=seg_data.get("text", ""),
            confidence=0.3,
        ))

    total_duration = max(s.end_time for s in segments) if segments else 0.0
    profiles = [SpeakerProfile(
        speaker_id="SPEAKER_00",
        total_duration=total_duration,
        segment_count=len(segments),
        speaking_ratio=1.0,
        avg_segment_duration=total_duration / max(len(segments), 1),
        sample_texts=[s.text[:200] for s in segments[:5] if s.text],
    )]

    return segments, profiles


# ---------------------------------------------------------------------------
# Emotion / sentiment analysis
# ---------------------------------------------------------------------------

def analyze_emotions(
    segments: List[SpeakerSegment],
    use_llm: bool = True,
    groq_api_key: Optional[str] = None,
) -> List[EmotionSegment]:
    """
    Analyze emotions and sentiment in speaker segments.

    Strategy:
    1. Try transformers-based emotion classification
    2. Fall back to keyword/lexicon-based analysis
    3. Optionally use LLM for nuanced spiritual/philosophical content

    Args:
        segments: Speaker segments with text
        use_llm: Whether to use LLM for emotion analysis
        groq_api_key: Groq API key for LLM-based analysis

    Returns:
        List of EmotionSegment objects
    """
    results: List[EmotionSegment] = []

    if not segments:
        return results

    # Try transformer-based emotion classification
    try:
        return _analyze_emotions_transformer(segments)
    except ImportError:
        logger.info("transformers not available for emotion analysis, using lexicon-based")
    except Exception as e:
        logger.warning(f"Transformer emotion analysis failed: {e}")

    # Fallback: lexicon/keyword-based emotion analysis
    return _analyze_emotions_lexicon(segments)


def _analyze_emotions_transformer(
    segments: List[SpeakerSegment],
) -> List[EmotionSegment]:
    """Emotion analysis using HuggingFace transformers pipeline."""
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=-1,  # CPU; use 0 for GPU
    )

    results: List[EmotionSegment] = []

    for seg in segments:
        if not seg.text or len(seg.text.strip()) < 5:
            continue

        try:
            # Truncate to model max length
            text = seg.text[:512]
            predictions = classifier(text)

            if predictions and isinstance(predictions[0], list):
                scores = {p["label"]: p["score"] for p in predictions[0]}
                primary = max(scores, key=scores.get)

                # Map to sentiment
                positive_emotions = {"joy", "surprise", "love"}
                negative_emotions = {"anger", "sadness", "fear", "disgust"}

                if primary in positive_emotions:
                    sentiment = "positive"
                    sentiment_score = scores.get(primary, 0.0)
                elif primary in negative_emotions:
                    sentiment = "negative"
                    sentiment_score = -scores.get(primary, 0.0)
                else:
                    sentiment = "neutral"
                    sentiment_score = 0.0

                results.append(EmotionSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=seg.text[:500],
                    primary_emotion=primary,
                    emotion_scores=scores,
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                ))
        except Exception as e:
            logger.warning(f"Emotion analysis failed for segment: {e}")

    return results


def _analyze_emotions_lexicon(
    segments: List[SpeakerSegment],
) -> List[EmotionSegment]:
    """
    Simple lexicon-based emotion analysis.
    Includes spiritual/philosophical emotion categories common in Sanskrit content.
    """
    # Emotion lexicon with common keywords
    lexicon = {
        "joy": {"happy", "joy", "wonderful", "beautiful", "love", "grateful", "blessed",
                "bliss", "आनन्द", "सुख", "प्रेम", "कृपा"},
        "devotion": {"devotion", "surrender", "divine", "sacred", "prayer", "worship",
                     "भक्ति", "प्रार्थना", "श्रद्धा", "पूजा", "ईश्वर"},
        "reverence": {"respect", "honor", "guru", "teacher", "wisdom", "noble",
                      "गुरु", "श्रद्धा", "सम्मान", "ज्ञान", "विद्या"},
        "peace": {"peace", "calm", "serene", "tranquil", "meditat", "silence",
                  "शान्ति", "ध्यान", "मौन", "समाधि"},
        "sadness": {"sad", "sorrow", "grief", "suffer", "pain", "loss",
                    "दुःख", "शोक", "पीड़ा", "कष्ट"},
        "anger": {"anger", "fury", "rage", "destroy", "fight", "war",
                  "क्रोध", "युद्ध", "संहार"},
        "fear": {"fear", "afraid", "terror", "danger", "death", "dark",
                 "भय", "मृत्यु", "अन्धकार"},
        "wonder": {"amazing", "miracle", "mysterious", "infinite", "cosmic",
                   "अद्भुत", "विस्मय", "अनन्त", "ब्रह्माण्ड"},
    }

    results: List[EmotionSegment] = []

    for seg in segments:
        if not seg.text or len(seg.text.strip()) < 5:
            continue

        text_lower = seg.text.lower()
        scores: Dict[str, float] = {}

        for emotion, keywords in lexicon.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            word_count = len(text_lower.split())
            scores[emotion] = count / max(word_count, 1)

        # Determine primary emotion
        if any(s > 0 for s in scores.values()):
            primary = max(scores, key=scores.get)
            confidence = scores[primary]
        else:
            primary = "neutral"
            confidence = 0.5
            scores["neutral"] = 0.5

        # Map to sentiment
        positive = {"joy", "devotion", "reverence", "peace", "wonder"}
        negative = {"sadness", "anger", "fear"}

        if primary in positive:
            sentiment = "positive"
            sentiment_score = confidence
        elif primary in negative:
            sentiment = "negative"
            sentiment_score = -confidence
        else:
            sentiment = "neutral"
            sentiment_score = 0.0

        results.append(EmotionSegment(
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=seg.text[:500],
            primary_emotion=primary,
            emotion_scores=scores,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
        ))

    return results


# ---------------------------------------------------------------------------
# Full speaker analysis pipeline
# ---------------------------------------------------------------------------

def analyze_speakers(
    audio_path: str,
    transcript_segments: Optional[List[Dict]] = None,
    max_speakers: int = 10,
    analyze_emotions_flag: bool = True,
) -> SpeakerAnalysis:
    """
    Complete speaker analysis pipeline.

    Args:
        audio_path: Path to audio file
        transcript_segments: Optional transcript segments with timing
        max_speakers: Maximum expected speakers
        analyze_emotions_flag: Whether to run emotion analysis

    Returns:
        SpeakerAnalysis with all speaker information
    """
    logger.info(f"Starting speaker analysis of {audio_path}")

    # Speaker diarization
    segments, profiles = identify_speakers(
        audio_path,
        transcript_segments=transcript_segments,
        max_speakers=max_speakers,
    )

    # Emotion analysis
    emotions: List[EmotionSegment] = []
    if analyze_emotions_flag and segments:
        emotions = analyze_emotions(segments)

    # Determine dominant speaker
    dominant = None
    if profiles:
        dominant = max(profiles, key=lambda p: p.total_duration).speaker_id

    total_duration = max(s.end_time for s in segments) if segments else 0.0

    return SpeakerAnalysis(
        speakers=profiles,
        segments=segments,
        emotions=emotions,
        total_speakers=len(profiles),
        total_duration=total_duration,
        dominant_speaker=dominant,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_speaker_profiles(
    speaker_data: Dict[str, Dict],
    total_duration: float,
) -> List[SpeakerProfile]:
    """Build speaker profiles from aggregated data."""
    profiles = []
    for speaker_id, data in speaker_data.items():
        profiles.append(SpeakerProfile(
            speaker_id=speaker_id,
            total_duration=data["duration"],
            segment_count=data["count"],
            speaking_ratio=data["duration"] / max(total_duration, 1.0),
            avg_segment_duration=data["duration"] / max(data["count"], 1),
            sample_texts=data["texts"][:5],
        ))
    return sorted(profiles, key=lambda p: p.total_duration, reverse=True)


def _align_text_to_segment(
    transcript_segments: List[Dict],
    start_time: float,
    end_time: float,
) -> str:
    """Align transcript text to a time range."""
    texts = []
    for seg in transcript_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        # Check overlap
        if seg_start < end_time and seg_end > start_time:
            texts.append(seg.get("text", ""))
    return " ".join(texts).strip()


def _detect_silence(
    audio_path: str,
    noise_threshold: str = "-30dB",
    min_silence_duration: float = 0.8,
) -> List[float]:
    """Detect silence boundaries in audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={noise_threshold}:d={min_silence_duration}",
        "-f", "null", "-",
        "-loglevel", "info",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        timestamps = []

        for line in result.stderr.split("\n"):
            # Parse silence_end timestamps
            match = re.search(r"silence_end:\s*(\d+\.?\d*)", line)
            if match:
                timestamps.append(float(match.group(1)))

        return timestamps

    except Exception as e:
        logger.warning(f"Silence detection failed: {e}")
        return []


def _get_audio_duration(audio_path: str) -> float:
    """Get audio file duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0
