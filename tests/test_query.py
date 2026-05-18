"""Tests for shruti.query — RAG query with mocked vectorstore and Groq."""

import pytest
from unittest.mock import patch, MagicMock


class TestQueryKnowledge:
    @patch("shruti.query.Groq")
    @patch("shruti.query.search_similar_chunks")
    @patch("shruti.query.get_config")
    def test_successful_query(self, mock_config, mock_search, mock_groq_cls, mock_groq_api_key):
        """query_knowledge returns answer with sources matching Source model."""
        cfg = MagicMock()
        cfg.GROQ_API_KEY = "test-key"
        cfg.LLM_MODEL = "llama-3.3-70b-versatile"
        mock_config.return_value = cfg

        mock_search.return_value = [
            {
                "text": "Dharma is righteousness.",
                "metadata": {"video_id": "v1", "title": "Dharma Talk", "chunk_index": 0},
                "similarity_score": 0.9,
            },
            {
                "text": "Karma relates to action.",
                "metadata": {"video_id": "v1", "title": "Dharma Talk", "chunk_index": 1, "timestamp": 60},
                "similarity_score": 0.7,
            },
        ]

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Dharma is the path of righteousness."
        mock_client.chat.completions.create.return_value = mock_resp

        from shruti.query import query_knowledge
        result = query_knowledge("What is Dharma?", top_k=5)

        assert result["answer"] == "Dharma is the path of righteousness."
        assert result["chunks_found"] == 2
        assert result["confidence"] in ("low", "medium", "high")
        assert "avg_similarity" in result

        # Verify sources match Source model fields
        for source in result["sources"]:
            assert "video_id" in source
            assert "title" in source
            assert "chunk_text" in source
            assert "similarity_score" in source
            assert "chunk_index" in source
            assert "timestamp" in source  # Can be None

    @patch("shruti.query.search_similar_chunks")
    @patch("shruti.query.get_config")
    def test_no_results(self, mock_config, mock_search, mock_groq_api_key):
        """Returns low confidence when no chunks found."""
        cfg = MagicMock()
        cfg.GROQ_API_KEY = "test-key"
        mock_config.return_value = cfg
        mock_search.return_value = []

        from shruti.query import query_knowledge
        result = query_knowledge("Unknown question")

        assert result["confidence"] == "low"
        assert result["chunks_found"] == 0
        assert result["sources"] == []
        assert result["avg_similarity"] == 0.0

    @patch("shruti.query.Groq")
    @patch("shruti.query.search_similar_chunks")
    @patch("shruti.query.get_config")
    def test_confidence_levels(self, mock_config, mock_search, mock_groq_cls, mock_groq_api_key):
        """Confidence is computed from avg similarity."""
        cfg = MagicMock()
        cfg.GROQ_API_KEY = "test-key"
        cfg.LLM_MODEL = "model"
        mock_config.return_value = cfg

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Answer"
        mock_client.chat.completions.create.return_value = mock_resp

        # High similarity
        mock_search.return_value = [
            {"text": "x", "metadata": {"video_id": "v", "title": "T", "chunk_index": 0}, "similarity_score": 0.9},
        ]
        from shruti.query import query_knowledge
        result = query_knowledge("q")
        assert result["confidence"] == "high"

        # Low similarity
        mock_search.return_value = [
            {"text": "x", "metadata": {"video_id": "v", "title": "T", "chunk_index": 0}, "similarity_score": 0.3},
        ]
        result = query_knowledge("q")
        assert result["confidence"] == "low"


class TestBatchQueryKnowledge:
    @patch("shruti.query.query_knowledge")
    def test_batch_processes_all(self, mock_query, mock_groq_api_key):
        """batch_query_knowledge processes all questions."""
        mock_query.return_value = {
            "answer": "test",
            "sources": [],
            "confidence": "medium",
            "chunks_found": 1,
            "avg_similarity": 0.6,
        }

        from shruti.query import batch_query_knowledge
        results = batch_query_knowledge(["Q1", "Q2", "Q3"])

        assert len(results) == 3
        assert all("question" in r for r in results)
        assert all("question_index" in r for r in results)

    @patch("shruti.query.query_knowledge")
    def test_batch_handles_errors(self, mock_query, mock_groq_api_key):
        """Batch handles per-question errors."""
        mock_query.side_effect = [
            {"answer": "ok", "sources": [], "confidence": "high", "chunks_found": 1, "avg_similarity": 0.8},
            Exception("fail"),
        ]

        from shruti.query import batch_query_knowledge
        results = batch_query_knowledge(["Q1", "Q2"])

        assert len(results) == 2
        assert "error" in results[1]


class TestGetRelatedTopics:
    @patch("shruti.query.search_similar_chunks")
    def test_returns_related_topic_dicts(self, mock_search, mock_groq_api_key):
        """get_related_topics returns RelatedTopic-compatible dicts."""
        mock_search.return_value = [
            {
                "text": "Dharma is Central to Understanding Vedanta philosophy.",
                "metadata": {"video_id": "v1", "title": "Introduction to Vedanta Philosophy"},
                "similarity_score": 0.8,
            },
        ]

        from shruti.query import get_related_topics
        results = get_related_topics("Dharma")

        assert isinstance(results, list)
        for item in results:
            assert "topic" in item
            assert "relevance_score" in item
            assert "video_count" in item

    @patch("shruti.query.search_similar_chunks")
    def test_empty_results(self, mock_search, mock_groq_api_key):
        """Returns empty list when no chunks found."""
        mock_search.return_value = []

        from shruti.query import get_related_topics
        results = get_related_topics("Unknown")
        assert results == []


class TestSearchByMetadata:
    @patch("shruti.query.load_vectorstore")
    def test_returns_metadata_search_results(self, mock_load, mock_groq_api_key):
        """search_by_metadata returns MetadataSearchResult-compatible dicts."""
        from langchain.docstore.document import Document

        mock_vs = MagicMock()
        doc = Document(
            page_content="test chunk",
            metadata={"video_id": "v1", "title": "T", "chunk_index": 2, "language": "en"},
        )
        mock_vs.similarity_search.return_value = [doc]
        mock_load.return_value = mock_vs

        from shruti.query import search_by_metadata
        results = search_by_metadata({"language": "en"})

        assert len(results) == 1
        assert results[0]["video_id"] == "v1"
        assert results[0]["chunk_text"] == "test chunk"
        assert results[0]["chunk_index"] == 2
        assert "metadata" in results[0]
