"""Tests for shruti.extract — knowledge extraction with mocked Groq."""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestExtractKnowledge:
    @patch("shruti.extract.Groq")
    @patch("shruti.extract.get_config")
    def test_successful_extraction(self, mock_config, mock_groq_cls, sample_knowledge, mock_groq_api_key):
        """extract_knowledge returns structured knowledge dict."""
        cfg = MagicMock()
        cfg.GROQ_API_KEY = "test-key"
        cfg.LLM_MODEL = "llama-3.3-70b-versatile"
        mock_config.return_value = cfg

        # Mock Groq response
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(sample_knowledge)
        mock_client.chat.completions.create.return_value = mock_response

        from shruti.extract import extract_knowledge
        result = extract_knowledge("Test transcript", "Test Title")

        assert "summary" in result
        assert "key_concepts" in result
        assert len(result["key_concepts"]) == 2
        assert "sanskrit_terms" in result

    @patch("shruti.extract.Groq")
    @patch("shruti.extract.get_config")
    def test_fallback_on_invalid_json(self, mock_config, mock_groq_cls, mock_groq_api_key):
        """Falls back to parsed response when JSON is invalid."""
        cfg = MagicMock()
        cfg.GROQ_API_KEY = "test-key"
        cfg.LLM_MODEL = "llama-3.3-70b-versatile"
        mock_config.return_value = cfg

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON {{"
        mock_client.chat.completions.create.return_value = mock_response

        from shruti.extract import extract_knowledge
        result = extract_knowledge("Test transcript", "Test Title")

        # Should return fallback structure
        assert "summary" in result
        assert "key_concepts" in result

    @patch("shruti.extract.Groq")
    @patch("shruti.extract.get_config")
    def test_missing_fields_get_defaults(self, mock_config, mock_groq_cls, mock_groq_api_key):
        """Missing fields in JSON response get default values."""
        cfg = MagicMock()
        cfg.GROQ_API_KEY = "test-key"
        cfg.LLM_MODEL = "llama-3.3-70b-versatile"
        mock_config.return_value = cfg

        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_response = MagicMock()
        # Only summary, missing other fields
        mock_response.choices[0].message.content = json.dumps({"summary": "Test"})
        mock_client.chat.completions.create.return_value = mock_response

        from shruti.extract import extract_knowledge
        result = extract_knowledge("Test transcript")

        assert result["summary"] == "Test"
        assert result["key_concepts"] == []
        assert isinstance(result["study_questions"], list)


class TestExtractKnowledgeBatch:
    @patch("shruti.extract.extract_knowledge")
    def test_batch_extraction(self, mock_extract, mock_groq_api_key):
        """extract_knowledge_batch processes multiple transcripts."""
        mock_extract.return_value = {"summary": "test", "key_concepts": []}

        from shruti.extract import extract_knowledge_batch
        transcripts = [
            {"text": "Transcript 1", "title": "Video 1"},
            {"text": "Transcript 2", "title": "Video 2"},
        ]
        results = extract_knowledge_batch(transcripts)

        assert len(results) == 2
        assert all("summary" in r for r in results)

    @patch("shruti.extract.extract_knowledge")
    def test_batch_handles_errors(self, mock_extract, mock_groq_api_key):
        """Batch extraction handles per-item errors gracefully."""
        mock_extract.side_effect = [
            {"summary": "ok"},
            Exception("API error"),
        ]

        from shruti.extract import extract_knowledge_batch
        transcripts = [
            {"text": "Good", "title": "V1"},
            {"text": "Bad", "title": "V2"},
        ]
        results = extract_knowledge_batch(transcripts)

        assert len(results) == 2
        assert "error" in results[1]


class TestFallbackParsing:
    def test_parse_fallback_response(self):
        """_parse_fallback_response extracts basic structure from text."""
        from shruti.extract import _parse_fallback_response

        text = """Summary: This is about Dharma.

Key Concepts:
- Dharma: The righteous path
- Karma: Action and consequence

Key Quotes:
- "Do your duty without attachment"

Study Questions:
- What is Dharma?
- How does Karma work?
"""
        result = _parse_fallback_response(text, "Test Title")

        assert "summary" in result
        assert isinstance(result["key_concepts"], list)
        assert isinstance(result["study_questions"], list)
