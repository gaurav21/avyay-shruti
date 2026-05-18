"""Tests for shruti.embeddings — chunking and vector storage with mocked deps."""

import pytest
from unittest.mock import patch, MagicMock
from langchain.docstore.document import Document


class TestChunkTranscript:
    @patch("shruti.embeddings.get_config")
    def test_basic_chunking(self, mock_config, sample_transcript, sample_metadata):
        """chunk_transcript produces Document objects with metadata."""
        cfg = MagicMock()
        cfg.CHUNK_SIZE = 200
        cfg.CHUNK_OVERLAP = 50
        mock_config.return_value = cfg

        from shruti.embeddings import chunk_transcript
        docs = chunk_transcript(sample_transcript, sample_metadata)

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # Each doc should have chunk_index in metadata
        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i
            assert "chunk_id" in doc.metadata
            assert doc.metadata["video_id"] == "dQw4w9WgXcQ"

    @patch("shruti.embeddings.get_config")
    def test_empty_text(self, mock_config):
        """Empty text produces no chunks."""
        cfg = MagicMock()
        cfg.CHUNK_SIZE = 500
        cfg.CHUNK_OVERLAP = 75
        mock_config.return_value = cfg

        from shruti.embeddings import chunk_transcript
        docs = chunk_transcript("", {"video_id": "x"})
        assert docs == []

    @patch("shruti.embeddings.get_config")
    def test_metadata_preserved(self, mock_config, sample_metadata):
        """Original metadata keys are preserved in each chunk."""
        cfg = MagicMock()
        cfg.CHUNK_SIZE = 500
        cfg.CHUNK_OVERLAP = 75
        mock_config.return_value = cfg

        from shruti.embeddings import chunk_transcript
        docs = chunk_transcript("Short text.", sample_metadata)

        if docs:
            assert docs[0].metadata["title"] == "Introduction to Dharma"
            assert docs[0].metadata["language"] == "en"


class TestGetEmbeddings:
    @patch("shruti.embeddings.HuggingFaceEmbeddings")
    @patch("shruti.embeddings.get_config")
    def test_returns_embeddings_instance(self, mock_config, mock_hf):
        """get_embeddings returns a HuggingFaceEmbeddings instance."""
        cfg = MagicMock()
        cfg.EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
        mock_config.return_value = cfg
        mock_hf.return_value = MagicMock()

        from shruti.embeddings import get_embeddings
        result = get_embeddings()

        mock_hf.assert_called_once()
        assert result is not None


class TestStoreChunks:
    @patch("shruti.embeddings.Chroma")
    @patch("shruti.embeddings.get_embeddings")
    @patch("shruti.embeddings.get_config")
    def test_store_chunks(self, mock_config, mock_embed, mock_chroma, tmp_path):
        """store_chunks creates vector store and adds documents."""
        cfg = MagicMock()
        cfg.get_chroma_persist_path.return_value = tmp_path / "chroma"
        mock_config.return_value = cfg

        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        docs = [
            Document(page_content="chunk 1", metadata={"chunk_id": "id1", "chunk_index": 0}),
            Document(page_content="chunk 2", metadata={"chunk_id": "id2", "chunk_index": 1}),
        ]

        from shruti.embeddings import store_chunks
        result = store_chunks(docs, str(tmp_path / "chroma"))

        mock_vs.add_texts.assert_called_once()
        assert result is mock_vs


class TestSearchSimilarChunks:
    @patch("shruti.embeddings.load_vectorstore")
    def test_search_returns_formatted(self, mock_load):
        """search_similar_chunks returns properly formatted results."""
        mock_vs = MagicMock()
        doc = Document(page_content="test text", metadata={"video_id": "v1"})
        mock_vs.similarity_search_with_score.return_value = [(doc, 0.85)]
        mock_load.return_value = mock_vs

        from shruti.embeddings import search_similar_chunks
        results = search_similar_chunks("query", k=5)

        assert len(results) == 1
        assert results[0]["text"] == "test text"
        assert results[0]["similarity_score"] == 0.85
        assert results[0]["metadata"]["video_id"] == "v1"


class TestGetDatabaseStats:
    @patch("shruti.embeddings.load_vectorstore")
    def test_stats(self, mock_load):
        """get_database_stats returns count and metadata."""
        mock_vs = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_vs._collection = mock_collection

        doc = Document(page_content="text", metadata={"video_id": "v1", "language": "en"})
        mock_vs.similarity_search.return_value = [doc]
        mock_load.return_value = mock_vs

        from shruti.embeddings import get_database_stats
        stats = get_database_stats("/tmp/test")

        assert stats["total_chunks"] == 42
        assert "en" in stats["languages"]

    @patch("shruti.embeddings.load_vectorstore")
    def test_stats_error(self, mock_load):
        """get_database_stats returns error dict on failure."""
        mock_load.side_effect = Exception("DB not found")

        from shruti.embeddings import get_database_stats
        stats = get_database_stats("/nonexistent")

        assert "error" in stats
