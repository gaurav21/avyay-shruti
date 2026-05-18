"""Tests for shruti.api — FastAPI endpoints with mocked dependencies."""

import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client(mock_groq_api_key):
    """Create a FastAPI TestClient with mocked startup."""
    with patch("shruti.api.validate_config"), \
         patch("shruti.api.setup_directories"), \
         patch("shruti.api.get_database_stats", return_value={
             "total_chunks": 0, "unique_videos": 0,
             "languages": [], "persist_dir": "/tmp/test",
         }):
        from fastapi.testclient import TestClient
        from shruti.api import app
        yield TestClient(app)


class TestRootAndHealth:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "ŚRUTI" in data["name"]
        assert "version" in data

    @patch("shruti.api.get_database_stats")
    @patch("shruti.api.validate_config")
    def test_health_healthy(self, mock_validate, mock_stats, client):
        mock_stats.return_value = {"total_chunks": 5}
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    @patch("shruti.api.get_database_stats", side_effect=Exception("no db"))
    @patch("shruti.api.validate_config")
    def test_health_unhealthy(self, mock_validate, mock_stats, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unhealthy"


class TestTranscribeEndpoint:
    @patch("shruti.api.transcribe_url")
    def test_transcribe_success(self, mock_transcribe, client):
        mock_transcribe.return_value = {
            "video_id": "abc",
            "title": "Test",
            "transcript": "Hello world",
            "language": "en",
            "source": "whisper",
            "url": "https://youtube.com/watch?v=abc",
            "duration": 300,
            "segments": [{"start": 0, "end": 5, "text": "Hello"}],
        }

        resp = client.post("/transcribe", json={"url": "https://youtube.com/watch?v=abc"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_id"] == "abc"
        assert data["url"] == "https://youtube.com/watch?v=abc"
        assert data["duration"] == 300

    @patch("shruti.api.transcribe_url", side_effect=Exception("yt-dlp error"))
    def test_transcribe_error(self, mock_transcribe, client):
        resp = client.post("/transcribe", json={"url": "https://youtube.com/watch?v=bad"})
        assert resp.status_code == 400


class TestExtractEndpoint:
    @patch("shruti.api.extract_knowledge")
    @patch("shruti.api.transcribe_url")
    def test_extract_success(self, mock_transcribe, mock_extract, client):
        mock_transcribe.return_value = {
            "video_id": "xyz",
            "title": "Extract Test",
            "transcript": "Some transcript",
            "language": "en",
            "source": "captions",
            "url": "https://youtube.com/watch?v=xyz",
            "duration": 100,
        }
        mock_extract.return_value = {"summary": "Test summary", "key_concepts": []}

        resp = client.post("/extract", json={"url": "https://youtube.com/watch?v=xyz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["knowledge"]["summary"] == "Test summary"


class TestQueryEndpoint:
    @patch("shruti.api.query_knowledge")
    def test_query_success(self, mock_query, client):
        mock_query.return_value = {
            "answer": "Dharma is...",
            "sources": [
                {
                    "video_id": "v1",
                    "title": "T",
                    "chunk_text": "text",
                    "similarity_score": 0.9,
                    "timestamp": None,
                    "chunk_index": 0,
                },
            ],
            "confidence": "high",
            "chunks_found": 1,
            "avg_similarity": 0.9,
        }

        resp = client.post("/query", json={"query": "What is Dharma?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Dharma is..."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["video_id"] == "v1"


class TestStatsEndpoint:
    @patch("shruti.api.get_database_stats")
    def test_stats(self, mock_stats, client):
        mock_stats.return_value = {
            "total_chunks": 100,
            "unique_videos": 5,
            "languages": ["en", "hi"],
            "persist_dir": "/tmp/test",
        }

        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] == 100

    @patch("shruti.api.get_database_stats")
    def test_stats_error(self, mock_stats, client):
        mock_stats.return_value = {"error": "DB not found"}

        resp = client.get("/stats")
        assert resp.status_code == 500


class TestRelatedTopicsEndpoint:
    @patch("shruti.api.get_related_topics")
    def test_related_topics(self, mock_related, client):
        mock_related.return_value = [
            {"topic": "Karma", "relevance_score": 0.8, "video_count": 2},
        ]

        resp = client.post("/related-topics", json={"topic": "Dharma"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["topic"] == "Dharma"
        assert len(data["related_topics"]) == 1


class TestMetadataSearchEndpoint:
    @patch("shruti.api.search_by_metadata")
    def test_metadata_search(self, mock_search, client):
        mock_search.return_value = [
            {
                "video_id": "v1",
                "title": "T",
                "chunk_text": "text",
                "chunk_index": 0,
                "metadata": {"language": "en"},
            },
        ]

        resp = client.post("/search/metadata", json={"filters": {"language": "en"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 1
