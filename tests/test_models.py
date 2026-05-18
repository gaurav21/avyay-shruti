"""Tests for shruti.models — Pydantic model validation."""

import pytest
from shruti.models import (
    HealthResponse,
    TranscribeRequest,
    TranscribeResponse,
    Segment,
    ExtractRequest,
    ExtractResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Source,
    BatchQueryRequest,
    BatchQueryResponse,
    DatabaseStatsResponse,
    RelatedTopic,
    RelatedTopicsRequest,
    RelatedTopicsResponse,
    MetadataSearchResult,
    MetadataSearchRequest,
    MetadataSearchResponse,
    ErrorResponse,
    ConfigInfo,
)


class TestHealthResponse:
    def test_valid(self):
        r = HealthResponse(status="healthy", version="0.1.0", config_valid=True, database_accessible=True)
        assert r.status == "healthy"

    def test_missing_field(self):
        with pytest.raises(Exception):
            HealthResponse(status="ok")


class TestTranscribeModels:
    def test_request(self):
        req = TranscribeRequest(url="https://youtube.com/watch?v=abc")
        assert req.url.startswith("https://")

    def test_response_minimal(self):
        resp = TranscribeResponse(
            video_id="abc",
            title="Test",
            transcript="Hello",
            language="en",
            source="whisper",
            url="https://youtube.com/watch?v=abc",
        )
        assert resp.duration is None
        assert resp.segments is None

    def test_response_with_segments(self):
        seg = Segment(start=0.0, end=5.0, text="hello")
        resp = TranscribeResponse(
            video_id="abc",
            title="Test",
            transcript="hello",
            language="en",
            source="whisper",
            url="https://youtube.com/watch?v=abc",
            duration=120.5,
            segments=[seg],
        )
        assert len(resp.segments) == 1
        assert resp.duration == 120.5


class TestSourceModel:
    def test_source_all_fields(self):
        s = Source(
            video_id="vid1",
            title="My Video",
            chunk_text="some text",
            similarity_score=0.9,
            timestamp=42.0,
            chunk_index=3,
        )
        assert s.video_id == "vid1"
        assert s.chunk_index == 3

    def test_source_optional_timestamp(self):
        s = Source(
            video_id="vid1",
            title="My Video",
            chunk_text="text",
            similarity_score=0.5,
            chunk_index=0,
        )
        assert s.timestamp is None


class TestQueryModels:
    def test_query_request_defaults(self):
        req = QueryRequest(query="What is Dharma?")
        assert req.top_k == 5
        assert req.temperature == 0.1
        assert req.include_metadata is True

    def test_query_response(self, sample_query_sources):
        resp = QueryResponse(
            answer="Dharma is...",
            sources=[Source(**s) for s in sample_query_sources],
            confidence="high",
            chunks_found=2,
            avg_similarity=0.78,
            query="What is Dharma?",
        )
        assert len(resp.sources) == 2
        assert resp.confidence == "high"

    def test_batch_query(self):
        qr = QueryResponse(
            answer="x", sources=[], confidence="low",
            chunks_found=0, avg_similarity=0.0, query="q",
        )
        batch = BatchQueryResponse(results=[qr], total_queries=1, successful_queries=1)
        assert batch.total_queries == 1


class TestExtractModels:
    def test_extract_request(self):
        req = ExtractRequest(url="https://youtube.com/watch?v=xyz")
        assert "xyz" in req.url

    def test_extract_response(self):
        resp = ExtractResponse(
            video_id="xyz",
            title="Test",
            knowledge={"summary": "test"},
            transcript_source="whisper",
            language="en",
        )
        assert resp.knowledge["summary"] == "test"


class TestIngestModels:
    def test_ingest_request_defaults(self):
        req = IngestRequest(url="https://youtube.com/watch?v=xyz")
        assert req.persist_dir is None

    def test_ingest_response(self):
        resp = IngestResponse(
            video_id="xyz",
            title="T",
            chunks_stored=10,
            language="hi",
            transcript_source="captions",
            persist_dir="/tmp/test",
        )
        assert resp.chunks_stored == 10


class TestDatabaseStatsResponse:
    def test_valid(self):
        resp = DatabaseStatsResponse(
            total_chunks=100,
            unique_videos=5,
            languages=["en", "hi"],
            persist_dir="/tmp",
        )
        assert resp.unique_videos == 5


class TestRelatedTopicModels:
    def test_related_topic(self):
        rt = RelatedTopic(topic="Karma", relevance_score=0.8, video_count=3)
        assert rt.topic == "Karma"

    def test_related_topics_response(self):
        resp = RelatedTopicsResponse(
            topic="Dharma",
            related_topics=[
                RelatedTopic(topic="Karma", relevance_score=0.8, video_count=2),
            ],
        )
        assert len(resp.related_topics) == 1


class TestMetadataSearchModels:
    def test_metadata_search_result(self):
        r = MetadataSearchResult(
            video_id="v1",
            title="T",
            chunk_text="text",
            chunk_index=0,
            metadata={"language": "en"},
        )
        assert r.metadata["language"] == "en"

    def test_metadata_search_response(self):
        resp = MetadataSearchResponse(
            results=[],
            total_found=0,
            filters_applied={"language": "en"},
        )
        assert resp.total_found == 0


class TestErrorAndConfigModels:
    def test_error_response(self):
        e = ErrorResponse(error="bad", details="details", error_type="ValueError")
        assert e.error_type == "ValueError"

    def test_config_info(self):
        c = ConfigInfo(
            embedding_model="model",
            llm_model="llama",
            chunk_size=500,
            chunk_overlap=75,
            chroma_persist_dir="/tmp",
        )
        assert c.chunk_size == 500
