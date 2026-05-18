"""
Shared fixtures for ŚRUTI tests.
All external services (Groq, yt-dlp, ChromaDB, HuggingFace) are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_groq_api_key(monkeypatch):
    """Set a fake GROQ_API_KEY so config validation passes."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-fake")


@pytest.fixture
def sample_transcript():
    """A sample transcript for testing."""
    return (
        "In this lecture we explore the concept of Dharma (धर्म). "
        "Dharma is a key concept in Indian philosophy. "
        "The Bhagavad Gita says: कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। "
        "This means you have the right to action but not to the fruits of action. "
        "We also discuss Karma (कर्म) and its relationship to Dharma. "
        "Sanskrit terms like Ātman (आत्मन्) and Brahman (ब्रह्मन्) are central "
        "to understanding Vedantic philosophy."
    )


@pytest.fixture
def sample_video_info():
    """Sample yt-dlp video info dict."""
    return {
        "id": "dQw4w9WgXcQ",
        "title": "Introduction to Dharma",
        "duration": 600,
        "subtitles": {},
        "automatic_captions": {},
    }


@pytest.fixture
def sample_video_info_with_captions():
    """Sample yt-dlp video info with captions."""
    return {
        "id": "abc123",
        "title": "Vedanta Lecture",
        "duration": 1200,
        "subtitles": {
            "en": [{"data": "Hello world. This is a test caption."}],
        },
        "automatic_captions": {},
    }


@pytest.fixture
def sample_metadata():
    """Sample video metadata for embeddings."""
    return {
        "video_id": "dQw4w9WgXcQ",
        "title": "Introduction to Dharma",
        "language": "en",
        "source": "whisper",
        "duration": 600,
        "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    }


@pytest.fixture
def sample_whisper_result():
    """Mock Groq Whisper transcription result."""
    result = MagicMock()
    result.text = "This is a transcribed text about Dharma and Karma."
    result.language = "en"
    
    seg1 = MagicMock()
    seg1.start = 0.0
    seg1.end = 5.0
    seg1.text = "This is a transcribed text"
    
    seg2 = MagicMock()
    seg2.start = 5.0
    seg2.end = 10.0
    seg2.text = "about Dharma and Karma."
    
    result.segments = [seg1, seg2]
    return result


@pytest.fixture
def sample_knowledge():
    """Sample extracted knowledge dict."""
    return {
        "summary": "A lecture about Dharma in Indian philosophy.",
        "key_concepts": [
            {"concept": "Dharma", "explanation": "Righteous duty"},
            {"concept": "Karma", "explanation": "Action and consequence"},
        ],
        "key_quotes": ["कर्मण्येवाधिकारस्ते मा फलेषु कदाचन"],
        "sanskrit_terms": [
            {"term": "धर्म", "meaning": "Righteousness", "transliteration": "Dharma"},
        ],
        "study_questions": ["What is Dharma?"] * 10,
        "follow_up_topics": ["Bhagavad Gita", "Vedanta", "Karma Yoga"],
    }


@pytest.fixture
def sample_query_sources():
    """Sample sources matching Source model."""
    return [
        {
            "video_id": "dQw4w9WgXcQ",
            "title": "Introduction to Dharma",
            "chunk_text": "Dharma is a key concept...",
            "similarity_score": 0.85,
            "timestamp": None,
            "chunk_index": 0,
        },
        {
            "video_id": "dQw4w9WgXcQ",
            "title": "Introduction to Dharma",
            "chunk_text": "Karma and Dharma are related...",
            "similarity_score": 0.72,
            "timestamp": 120.0,
            "chunk_index": 1,
        },
    ]
