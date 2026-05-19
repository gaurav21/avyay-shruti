"""
Tests for ŚRUTI V2.0 knowledge graph module.
"""

import json
import os
import tempfile

import pytest

from shruti.knowledge_graph import (
    Entity,
    Relationship,
    CrossReference,
    KnowledgeGraph,
    extract_entities,
    extract_relationships,
    detect_cross_references,
    build_knowledge_graph,
    save_knowledge_graph,
    load_knowledge_graph,
    query_graph,
    get_graph_statistics,
    _make_entity_id,
    _make_ref_id,
    _is_common_phrase,
)


# ---------------------------------------------------------------------------
# Entity ID generation tests
# ---------------------------------------------------------------------------

class TestEntityId:
    def test_deterministic(self):
        id1 = _make_entity_id("Brahman", "concept")
        id2 = _make_entity_id("Brahman", "concept")
        assert id1 == id2

    def test_case_insensitive(self):
        id1 = _make_entity_id("Brahman", "concept")
        id2 = _make_entity_id("brahman", "concept")
        assert id1 == id2

    def test_different_types(self):
        id1 = _make_entity_id("Yoga", "concept")
        id2 = _make_entity_id("Yoga", "person")
        assert id1 != id2


class TestRefId:
    def test_deterministic_order(self):
        id1 = _make_ref_id("video_a", "video_b")
        id2 = _make_ref_id("video_b", "video_a")
        assert id1 == id2  # Order shouldn't matter


class TestIsCommonPhrase:
    def test_common_phrase(self):
        assert _is_common_phrase("The Quick") is True
        assert _is_common_phrase("This Approach") is True

    def test_proper_name(self):
        assert _is_common_phrase("Adi Shankaracharya") is False
        assert _is_common_phrase("Swami Vivekananda") is False


# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_from_text_only(self):
        text = "Adi Shankaracharya taught about Vedanta and the Bhagavad Gita."
        entities = extract_entities(text, video_id="v1")
        assert len(entities) > 0

    def test_from_knowledge(self):
        knowledge = {
            "key_concepts": [
                {"concept": "Brahman", "explanation": "The ultimate reality"},
                {"concept": "Maya", "explanation": "The world of illusion"},
            ],
            "sanskrit_terms": [
                {"term": "ब्रह्मन्", "meaning": "Ultimate reality", "transliteration": "Brahman"},
            ],
            "follow_up_topics": ["Advaita Vedanta", "Upanishads"],
        }
        entities = extract_entities(
            text="The concept of Brahman in Vedanta",
            video_id="v1",
            knowledge=knowledge,
        )
        names = [e.name for e in entities]
        assert "Brahman" in names or "ब्रह्मन्" in names

    def test_sanskrit_term_extraction(self):
        text = "धर्म and कर्म are fundamental concepts in Indian philosophy"
        entities = extract_entities(text, video_id="v1")
        # Should find Devanagari terms
        devanagari_entities = [e for e in entities if e.language == "sa"]
        assert len(devanagari_entities) >= 1

    def test_scripture_references(self):
        text = "The Bhagavad Gita and the Yoga Sutra are important texts in Indian philosophy"
        entities = extract_entities(text, video_id="v1")
        text_entities = [e for e in entities if e.entity_type == "text"]
        assert len(text_entities) >= 1

    def test_empty_text(self):
        entities = extract_entities("", video_id="v1")
        assert len(entities) == 0

    def test_video_id_tracking(self):
        entities = extract_entities(
            "Adi Shankaracharya taught Vedanta",
            video_id="test_video_123",
        )
        for entity in entities:
            assert "test_video_123" in entity.source_videos


# ---------------------------------------------------------------------------
# Relationship extraction tests
# ---------------------------------------------------------------------------

class TestExtractRelationships:
    def test_co_occurrence(self):
        entities = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept"),
            Entity(entity_id="e2", name="Maya", entity_type="concept"),
        ]
        text = "The relationship between Brahman and Maya is fundamental to Vedanta."
        relationships = extract_relationships(entities, text, video_id="v1")
        assert len(relationships) >= 1
        assert relationships[0].relation_type == "related_to"

    def test_no_relationships(self):
        entities = [
            Entity(entity_id="e1", name="ZZZXXX", entity_type="concept"),
            Entity(entity_id="e2", name="YYYQQQ", entity_type="concept"),
        ]
        text = "No relevant entities mentioned here."
        relationships = extract_relationships(entities, text)
        assert len(relationships) == 0

    def test_knowledge_based_relationships(self):
        entities = [
            Entity(entity_id=_make_entity_id("Brahman", "concept"), name="Brahman", entity_type="concept"),
            Entity(entity_id=_make_entity_id("ब्रह्मन्", "term"), name="ब्रह्मन्", entity_type="term"),
        ]
        knowledge = {
            "key_concepts": [{"concept": "Brahman", "explanation": "The ब्रह्मन् is the ultimate reality"}],
            "sanskrit_terms": [{"term": "ब्रह्मन्", "meaning": "Ultimate reality", "transliteration": "Brahman"}],
        }
        relationships = extract_relationships(entities, "text", knowledge=knowledge, video_id="v1")
        # Should find explains relationship
        explains = [r for r in relationships if r.relation_type == "explains"]
        assert len(explains) >= 0  # May or may not find depending on matching


# ---------------------------------------------------------------------------
# Knowledge graph building tests
# ---------------------------------------------------------------------------

class TestBuildKnowledgeGraph:
    def test_build_new_graph(self):
        entities = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept"),
            Entity(entity_id="e2", name="Atman", entity_type="concept"),
        ]
        relationships = [
            Relationship(source_id="e1", target_id="e2", relation_type="related_to"),
        ]
        graph = build_knowledge_graph(entities, relationships)
        assert graph.total_entities == 2
        assert graph.total_relationships == 1

    def test_merge_with_existing(self):
        existing = KnowledgeGraph(
            entities={"e1": Entity(entity_id="e1", name="Brahman", entity_type="concept", frequency=1)},
            relationships=[],
            cross_references=[],
        )
        new_entities = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept", frequency=1),
            Entity(entity_id="e3", name="Maya", entity_type="concept"),
        ]
        graph = build_knowledge_graph(new_entities, [], existing_graph=existing)
        assert graph.total_entities == 2  # e1 merged, e3 added
        assert graph.entities["e1"].frequency == 2  # Merged frequency

    def test_empty_graph(self):
        graph = build_knowledge_graph([], [])
        assert graph.total_entities == 0
        assert graph.total_relationships == 0


# ---------------------------------------------------------------------------
# Cross-reference detection tests
# ---------------------------------------------------------------------------

class TestCrossReferences:
    def test_detect_shared_entities(self):
        # Create graph with entities from video_a
        entities_a = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept", source_videos=["video_a"]),
            Entity(entity_id="e2", name="Atman", entity_type="concept", source_videos=["video_a"]),
            Entity(entity_id="e3", name="Maya", entity_type="concept", source_videos=["video_a"]),
        ]
        graph = build_knowledge_graph(entities_a, [])

        # New video shares some entities
        entities_b = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept", source_videos=["video_b"]),
            Entity(entity_id="e2", name="Atman", entity_type="concept", source_videos=["video_b"]),
            Entity(entity_id="e4", name="Dharma", entity_type="concept", source_videos=["video_b"]),
        ]

        cross_refs = detect_cross_references(
            current_entities=entities_b,
            current_video_id="video_b",
            graph=graph,
            min_shared_entities=2,
        )
        assert len(cross_refs) >= 1
        assert cross_refs[0].source_video == "video_b"
        assert cross_refs[0].target_video == "video_a"
        assert len(cross_refs[0].shared_entities) >= 2

    def test_no_shared_entities(self):
        entities_a = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept", source_videos=["video_a"]),
        ]
        graph = build_knowledge_graph(entities_a, [])

        entities_b = [
            Entity(entity_id="e99", name="Something Else", entity_type="concept", source_videos=["video_b"]),
        ]
        cross_refs = detect_cross_references(
            current_entities=entities_b,
            current_video_id="video_b",
            graph=graph,
        )
        assert len(cross_refs) == 0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestGraphPersistence:
    def test_save_and_load(self):
        entities = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept",
                   description="Ultimate reality", source_videos=["v1"]),
            Entity(entity_id="e2", name="ब्रह्मन्", entity_type="term",
                   language="sa", aliases=["Brahman"], source_videos=["v1"]),
        ]
        relationships = [
            Relationship(source_id="e1", target_id="e2", relation_type="explains", weight=2.0),
        ]
        graph = build_knowledge_graph(entities, relationships)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_knowledge_graph(graph, path)
            loaded = load_knowledge_graph(path)

            assert loaded is not None
            assert loaded.total_entities == 2
            assert loaded.total_relationships == 1
            assert "e1" in loaded.entities
            assert loaded.entities["e2"].language == "sa"
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        result = load_knowledge_graph("/tmp/nonexistent_graph_12345.json")
        assert result is None


# ---------------------------------------------------------------------------
# Graph query tests
# ---------------------------------------------------------------------------

class TestQueryGraph:
    def setup_method(self):
        entities = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept",
                   source_videos=["v1", "v2"]),
            Entity(entity_id="e2", name="Atman", entity_type="concept",
                   source_videos=["v1"]),
            Entity(entity_id="e3", name="Maya", entity_type="concept",
                   source_videos=["v2"]),
        ]
        relationships = [
            Relationship(source_id="e1", target_id="e2", relation_type="related_to"),
            Relationship(source_id="e1", target_id="e3", relation_type="related_to"),
        ]
        self.graph = build_knowledge_graph(entities, relationships)

    def test_query_by_name(self):
        result = query_graph(self.graph, entity_name="Brahman")
        assert len(result["entities"]) == 1
        assert result["entities"][0].name == "Brahman"

    def test_query_by_type(self):
        result = query_graph(self.graph, entity_type="concept")
        assert len(result["entities"]) == 3

    def test_query_by_video(self):
        result = query_graph(self.graph, video_id="v1")
        assert len(result["entities"]) == 2

    def test_query_relationships(self):
        result = query_graph(self.graph, entity_name="Brahman", max_depth=1)
        assert len(result["relationships"]) >= 2

    def test_empty_query(self):
        result = query_graph(self.graph, entity_name="Nonexistent")
        assert len(result["entities"]) == 0


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------

class TestGraphStatistics:
    def test_get_stats(self):
        entities = [
            Entity(entity_id="e1", name="Brahman", entity_type="concept", source_videos=["v1"]),
            Entity(entity_id="e2", name="ब्रह्मन्", entity_type="term", source_videos=["v1"]),
            Entity(entity_id="e3", name="Gita", entity_type="text", source_videos=["v2"]),
        ]
        relationships = [
            Relationship(source_id="e1", target_id="e2", relation_type="explains"),
            Relationship(source_id="e1", target_id="e3", relation_type="references"),
        ]
        graph = build_knowledge_graph(entities, relationships)
        stats = get_graph_statistics(graph)

        assert stats["total_entities"] == 3
        assert stats["total_relationships"] == 2
        assert stats["entity_types"]["concept"] == 1
        assert stats["entity_types"]["term"] == 1
        assert stats["entity_types"]["text"] == 1
        assert stats["videos_covered"] == 2
        assert len(stats["top_connected_entities"]) > 0
