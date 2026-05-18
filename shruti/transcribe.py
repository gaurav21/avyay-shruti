"""
YouTube video transcription module using yt-dlp and Groq Whisper.
"""

import os
import tempfile
import yt_dlp
from groq import Groq
from .config import config


def download_audio(url: str) -> str:
    """Download audio from YouTube URL using yt-dlp."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return output_path


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Groq Whisper API."""
    client = Groq(api_key=config.GROQ_API_KEY)
    
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model=config.WHISPER_MODEL,
            response_format="json",
            language="auto"
        )
    
    return transcription


def extract_captions(url: str) -> dict:
    """Extract existing captions from YouTube video."""
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'hi', 'sa'],
        'skip_download': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info


def _extract_caption_text(caption_info: dict) -> str:
    """Extract actual caption text from yt-dlp subtitle info."""
    subtitles = caption_info.get('subtitles', {})
    automatic_captions = caption_info.get('automatic_captions', {})
    all_subs = {**subtitles, **automatic_captions}

    # Prefer manual subtitles, then auto captions
    # Priority: en > hi > sa > first available
    for lang in ['en', 'hi', 'sa']:
        if lang in subtitles:
            return _get_text_from_sub_entries(subtitles[lang]), lang
        if lang in automatic_captions:
            return _get_text_from_sub_entries(automatic_captions[lang]), lang

    # Fall back to first available
    if all_subs:
        lang = list(all_subs.keys())[0]
        return _get_text_from_sub_entries(all_subs[lang]), lang

    return "", "unknown"


def _get_text_from_sub_entries(entries: list) -> str:
    """Extract plain text from subtitle format entries."""
    # yt-dlp returns a list of format dicts, each with 'url' and 'ext'
    # For formats like json3/srv3, we'd need to download and parse.
    # For simple cases, entries may contain text fragments.
    text_parts = []
    for entry in entries:
        if isinstance(entry, dict):
            # If there's direct text content
            if 'data' in entry:
                text_parts.append(entry['data'])
            elif 'text' in entry:
                text_parts.append(entry['text'])
    return ' '.join(text_parts) if text_parts else ""


def transcribe_url(url: str) -> dict:
    """Transcribe YouTube video from URL."""
    try:
        # Get video info
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id', '')
            title = info.get('title', '')
        
        # Try existing captions first
        caption_info = extract_captions(url)
        subtitles = caption_info.get('subtitles', {})
        automatic_captions = caption_info.get('automatic_captions', {})
        
        duration = info.get('duration')  # Duration in seconds
        segments = None

        if subtitles or automatic_captions:
            # Use existing captions if available
            caption_text, lang = _extract_caption_text(caption_info)
            if caption_text:
                transcript = caption_text
            else:
                # Captions exist but couldn't extract text — fall back to Whisper
                audio_path = download_audio(url)
                try:
                    result = transcribe_audio(audio_path)
                    transcript = result.text
                    lang = result.language or "unknown"
                    segments = _parse_whisper_segments(result)
                finally:
                    os.unlink(audio_path)
            source = "captions" if caption_text else "whisper"
        else:
            # Download and transcribe audio
            audio_path = download_audio(url)
            try:
                result = transcribe_audio(audio_path)
                transcript = result.text
                lang = result.language or "unknown"
                source = "whisper"
                segments = _parse_whisper_segments(result)
            finally:
                os.unlink(audio_path)  # Clean up
        
        return {
            "video_id": video_id,
            "title": title,
            "transcript": transcript,
            "language": lang,
            "source": source,
            "url": url,
            "duration": duration,
            "segments": segments,
        }
    
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")


def _parse_whisper_segments(result) -> list:
    """Parse segments from Whisper transcription result."""
    segments = []
    if hasattr(result, 'segments') and result.segments:
        for seg in result.segments:
            segments.append({
                "start": getattr(seg, 'start', 0.0),
                "end": getattr(seg, 'end', 0.0),
                "text": getattr(seg, 'text', ''),
            })
    return segments if segments else None