"""
Knowledge graph integration module for ŚRUTI V2.0.
Builds and manages a knowledge graph linking concepts, entities, and relationships
extracted from video content. Supports cross-reference linking between videos.
"""

import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity/concept in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: str  # concept, person, text, term, place, event
    description: str = ""
    language: str = "en"
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_videos: List[str] = field(default_factory=list)  # video IDs
    frequency: int = 1


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: str  # related_to, part_of, teaches, quotes, references, explains
    weight: float = 1.0
    context: str = ""
    source_video: str = ""


@dataclass
class CrossReference:
    """A cross-reference between content in different videos."""
    ref_id: str
    source_video: str
    target_video: str
    source_context: str  # text from source
    target_context: str  # text from target
    shared_entities: List[str]
    similarity_score: float = 0.0
    ref_type: str = "topic_overlap"  # topic_overlap, entity_shared, concept_continuation


@dataclass
class KnowledgeGraph:
    """The complete knowledge graph structure."""
    entities: Dict[str, Entity]  # entity_id -> Entity
    relationships: List[Relationship]
    cross_references: List[CrossReference]
    total_entities: int = 0
    total_relationships: int = 0
    total_cross_references: int = 0
    graph_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(
    text: str,
    video_id: str = "",
    title: str = "",
    knowledge: Optional[Dict] = None,
    language: str = "en",
) -> List[Entity]:
    """
    Extract named entities and concepts from text.

    Sources:
    1. Structured knowledge (from extract.py) — key concepts, Sanskrit terms
    2. NER (Named Entity Recognition) from text
    3. Pattern-based extraction for Sanskrit terms

    Args:
        text: Transcript or content text
        video_id: Source video identifier
        title: Video title
        knowledge: Structured knowledge dict from extract_knowledge()
        language: Primary language of the text

    Returns:
        List of Entity objects
    """
    entities: Dict[str, Entity] = {}

    # Extract from structured knowledge if available
    if knowledge:
        _extract_from_knowledge(knowledge, video_id, entities)

    # Extract Sanskrit terms from text
    _extract_sanskrit_terms(text, video_id, entities)

    # Extract proper nouns and concepts via patterns
    _extract_named_entities_pattern(text, video_id, entities)

    # Try NER with spaCy if available
    try:
        _extract_ner_spacy(text, video_id, entities)
    except ImportError:
        logger.debug("spaCy not available for NER, using pattern-based extraction only")

    result = list(entities.values())
    logger.info(f"Extracted {len(result)} entities from text ({video_id})")
    return result


def _extract_from_knowledge(
    knowledge: Dict,
    video_id: str,
    entities: Dict[str, Entity],
) -> None:
    """Extract entities from structured knowledge output."""
    # Key concepts
    for concept in knowledge.get("key_concepts", []):
        name = concept.get("concept", "")
        if not name:
            continue
        eid = _make_entity_id(name, "concept")
        if eid in entities:
            entities[eid].frequency += 1
            if video_id not in entities[eid].source_videos:
                entities[eid].source_videos.append(video_id)
        else:
            entities[eid] = Entity(
                entity_id=eid,
                name=name,
                entity_type="concept",
                description=concept.get("explanation", ""),
                source_videos=[video_id],
            )

    # Sanskrit terms
    for term in knowledge.get("sanskrit_terms", []):
        name = term.get("term", "")
        if not name:
            continue
        eid = _make_entity_id(name, "term")
        transliteration = term.get("transliteration", "")
        meaning = term.get("meaning", "")

        if eid in entities:
            entities[eid].frequency += 1
            if video_id not in entities[eid].source_videos:
                entities[eid].source_videos.append(video_id)
        else:
            entities[eid] = Entity(
                entity_id=eid,
                name=name,
                entity_type="term",
                description=meaning,
                language="sa",
                aliases=[transliteration] if transliteration else [],
                properties={
                    "transliteration": transliteration,
                    "meaning": meaning,
                },
                source_videos=[video_id],
            )

    # Follow-up topics
    for topic in knowledge.get("follow_up_topics", []):
        if not topic:
            continue
        eid = _make_entity_id(topic, "concept")
        if eid not in entities:
            entities[eid] = Entity(
                entity_id=eid,
                name=topic,
                entity_type="concept",
                description=f"Related topic: {topic}",
                source_videos=[video_id],
            )


def _extract_sanskrit_terms(
    text: str,
    video_id: str,
    entities: Dict[str, Entity],
) -> None:
    """Extract Sanskrit/Devanagari terms from text."""
    # Match Devanagari words (3+ characters)
    devanagari_pattern = r'[\u0900-\u097F]{3,}'
    matches = re.findall(devanagari_pattern, text)

    for term in set(matches):
        eid = _make_entity_id(term, "term")
        if eid in entities:
            entities[eid].frequency += 1
        else:
            entities[eid] = Entity(
                entity_id=eid,
                name=term,
                entity_type="term",
                language="sa",
                source_videos=[video_id],
            )


def _extract_named_entities_pattern(
    text: str,
    video_id: str,
    entities: Dict[str, Entity],
) -> None:
    """Pattern-based named entity extraction."""
    # Capitalized multi-word names (likely proper nouns)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    for match in re.finditer(name_pattern, text):
        name = match.group(1)
        if len(name) > 5 and not _is_common_phrase(name):
            eid = _make_entity_id(name, "person")
            if eid in entities:
                entities[eid].frequency += 1
            else:
                entities[eid] = Entity(
                    entity_id=eid,
                    name=name,
                    entity_type="person",
                    source_videos=[video_id],
                )

    # Scripture/text references
    text_refs = [
        r'\b(Bhagavad\s+Gita|Gita)\b',
        r'\b(Upanishad|Vedanta|Yoga\s+Sutra|Brahma\s+Sutra)\b',
        r'\b(Rig\s+Veda|Sama\s+Veda|Yajur\s+Veda|Atharva\s+Veda)\b',
        r'\b(Ramayana|Mahabharata|Purana)\b',
        r'\b(Patanjali|Shankaracharya|Vivekananda)\b',
    ]
    for pattern in text_refs:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(1)
            eid = _make_entity_id(name, "text")
            if eid in entities:
                entities[eid].frequency += 1
            else:
                entities[eid] = Entity(
                    entity_id=eid,
                    name=name,
                    entity_type="text",
                    description=f"Referenced text/scripture: {name}",
                    source_videos=[video_id],
                )


def _extract_ner_spacy(
    text: str,
    video_id: str,
    entities: Dict[str, Entity],
) -> None:
    """Extract entities using spaCy NER."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("spaCy en_core_web_sm not found, skipping NER")
        return

    # Process in chunks to handle long texts
    max_length = 100000
    doc = nlp(text[:max_length])

    type_map = {
        "PERSON": "person",
        "ORG": "concept",
        "GPE": "place",
        "LOC": "place",
        "EVENT": "event",
        "WORK_OF_ART": "text",
        "NORP": "concept",  # nationalities, religious groups
    }

    for ent in doc.ents:
        if ent.label_ not in type_map:
            continue

        name = ent.text.strip()
        if len(name) < 3:
            continue

        entity_type = type_map[ent.label_]
        eid = _make_entity_id(name, entity_type)

        if eid in entities:
            entities[eid].frequency += 1
        else:
            entities[eid] = Entity(
                entity_id=eid,
                name=name,
                entity_type=entity_type,
                source_videos=[video_id],
            )


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------

def extract_relationships(
    entities: List[Entity],
    text: str,
    knowledge: Optional[Dict] = None,
    video_id: str = "",
) -> List[Relationship]:
    """
    Extract relationships between entities from text and knowledge.

    Strategies:
    1. Co-occurrence in text (entities mentioned near each other)
    2. Knowledge structure (concepts related via key_concepts)
    3. Cross-reference patterns (entity A references entity B)

    Returns:
        List of Relationship objects
    """
    relationships: List[Relationship] = []
    entity_by_name: Dict[str, Entity] = {}

    for entity in entities:
        entity_by_name[entity.name.lower()] = entity
        for alias in entity.aliases:
            entity_by_name[alias.lower()] = entity

    # Co-occurrence relationships
    sentences = re.split(r'[.!?।॥]\s+', text)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        found_entities: List[Entity] = []

        for name, entity in entity_by_name.items():
            if name in sentence_lower:
                found_entities.append(entity)

        # Create pairwise relationships for co-occurring entities
        for i in range(len(found_entities)):
            for j in range(i + 1, len(found_entities)):
                e1, e2 = found_entities[i], found_entities[j]
                if e1.entity_id == e2.entity_id:
                    continue

                relationships.append(Relationship(
                    source_id=e1.entity_id,
                    target_id=e2.entity_id,
                    relation_type="related_to",
                    weight=1.0,
                    context=sentence[:200],
                    source_video=video_id,
                ))

    # Knowledge-based relationships
    if knowledge:
        _add_knowledge_relationships(knowledge, entities, relationships, video_id)

    # Deduplicate and merge weights
    relationships = _merge_relationships(relationships)

    logger.info(f"Extracted {len(relationships)} relationships")
    return relationships


def _add_knowledge_relationships(
    knowledge: Dict,
    entities: List[Entity],
    relationships: List[Relationship],
    video_id: str,
) -> None:
    """Add relationships based on structured knowledge."""
    entity_map = {e.name.lower(): e for e in entities}

    # Connect concepts to Sanskrit terms that appear in same content
    concepts = knowledge.get("key_concepts", [])
    terms = knowledge.get("sanskrit_terms", [])

    for concept in concepts:
        concept_name = concept.get("concept", "").lower()
        concept_entity = entity_map.get(concept_name)
        if not concept_entity:
            continue

        for term in terms:
            term_name = term.get("term", "").lower()
            term_entity = entity_map.get(term_name)
            if not term_entity:
                continue

            # Check if term is mentioned in concept explanation
            explanation = concept.get("explanation", "").lower()
            if term_name in explanation or term.get("transliteration", "").lower() in explanation:
                relationships.append(Relationship(
                    source_id=concept_entity.entity_id,
                    target_id=term_entity.entity_id,
                    relation_type="explains",
                    weight=2.0,
                    context=f"{concept_entity.name} relates to {term_entity.name}",
                    source_video=video_id,
                ))


def _merge_relationships(relationships: List[Relationship]) -> List[Relationship]:
    """Merge duplicate relationships, summing weights."""
    merged: Dict[Tuple[str, str, str], Relationship] = {}

    for rel in relationships:
        key = (rel.source_id, rel.target_id, rel.relation_type)
        if key in merged:
            merged[key].weight += rel.weight
        else:
            merged[key] = rel

    return list(merged.values())


# ---------------------------------------------------------------------------
# Cross-reference detection
# ---------------------------------------------------------------------------

def detect_cross_references(
    current_entities: List[Entity],
    current_video_id: str,
    graph: KnowledgeGraph,
    min_shared_entities: int = 2,
    min_similarity: float = 0.3,
) -> List[CrossReference]:
    """
    Detect cross-references between the current video and existing graph content.

    Finds connections by:
    1. Shared entities between videos
    2. Similar topic coverage
    3. Concept continuations

    Args:
        current_entities: Entities from current video
        current_video_id: Current video ID
        graph: Existing knowledge graph
        min_shared_entities: Minimum shared entities for a cross-reference
        min_similarity: Minimum Jaccard similarity threshold

    Returns:
        List of CrossReference objects
    """
    cross_refs: List[CrossReference] = []

    # Build video -> entity mapping from existing graph
    video_entities: Dict[str, Set[str]] = defaultdict(set)
    for entity in graph.entities.values():
        for vid in entity.source_videos:
            video_entities[vid].add(entity.entity_id)

    # Current video's entity set
    current_entity_ids = {e.entity_id for e in current_entities}
    current_entity_names = {e.name for e in current_entities}

    # Find cross-references with each other video
    for other_video_id, other_entity_ids in video_entities.items():
        if other_video_id == current_video_id:
            continue

        shared_ids = current_entity_ids & other_entity_ids
        if len(shared_ids) < min_shared_entities:
            continue

        # Calculate Jaccard similarity
        union = current_entity_ids | other_entity_ids
        similarity = len(shared_ids) / max(len(union), 1)

        if similarity < min_similarity:
            continue

        # Get shared entity names
        shared_names = []
        for eid in shared_ids:
            if eid in graph.entities:
                shared_names.append(graph.entities[eid].name)

        ref_id = _make_ref_id(current_video_id, other_video_id)

        # Determine reference type
        if similarity > 0.7:
            ref_type = "concept_continuation"
        elif len(shared_ids) > 5:
            ref_type = "topic_overlap"
        else:
            ref_type = "entity_shared"

        cross_refs.append(CrossReference(
            ref_id=ref_id,
            source_video=current_video_id,
            target_video=other_video_id,
            source_context=f"Contains {len(current_entity_ids)} entities",
            target_context=f"Contains {len(other_entity_ids)} entities",
            shared_entities=shared_names[:20],
            similarity_score=similarity,
            ref_type=ref_type,
        ))

    logger.info(f"Found {len(cross_refs)} cross-references for video {current_video_id}")
    return cross_refs


# ---------------------------------------------------------------------------
# Knowledge graph management
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    entities: List[Entity],
    relationships: List[Relationship],
    cross_references: Optional[List[CrossReference]] = None,
    existing_graph: Optional[KnowledgeGraph] = None,
) -> KnowledgeGraph:
    """
    Build or update a knowledge graph.

    Args:
        entities: New entities to add
        relationships: New relationships to add
        cross_references: New cross-references to add
        existing_graph: Existing graph to merge into

    Returns:
        Updated KnowledgeGraph
    """
    if existing_graph:
        graph_entities = dict(existing_graph.entities)
        graph_relationships = list(existing_graph.relationships)
        graph_cross_refs = list(existing_graph.cross_references)
    else:
        graph_entities = {}
        graph_relationships = []
        graph_cross_refs = []

    # Merge entities
    for entity in entities:
        if entity.entity_id in graph_entities:
            existing = graph_entities[entity.entity_id]
            existing.frequency += entity.frequency
            for vid in entity.source_videos:
                if vid not in existing.source_videos:
                    existing.source_videos.append(vid)
            # Update description if current is longer
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
        else:
            graph_entities[entity.entity_id] = entity

    # Add relationships (merge duplicates)
    graph_relationships.extend(relationships)
    graph_relationships = _merge_relationships(graph_relationships)

    # Add cross-references
    if cross_references:
        existing_ref_ids = {cr.ref_id for cr in graph_cross_refs}
        for cr in cross_references:
            if cr.ref_id not in existing_ref_ids:
                graph_cross_refs.append(cr)

    graph = KnowledgeGraph(
        entities=graph_entities,
        relationships=graph_relationships,
        cross_references=graph_cross_refs,
        total_entities=len(graph_entities),
        total_relationships=len(graph_relationships),
        total_cross_references=len(graph_cross_refs),
        graph_metadata={
            "last_updated": _now_iso(),
            "entity_types": _count_by_type(graph_entities),
        },
    )

    logger.info(
        f"Knowledge graph: {graph.total_entities} entities, "
        f"{graph.total_relationships} relationships, "
        f"{graph.total_cross_references} cross-references"
    )
    return graph


def save_knowledge_graph(graph: KnowledgeGraph, path: str) -> None:
    """Save knowledge graph to JSON file."""
    data = {
        "entities": {
            eid: {
                "entity_id": e.entity_id,
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "language": e.language,
                "aliases": e.aliases,
                "properties": e.properties,
                "source_videos": e.source_videos,
                "frequency": e.frequency,
            }
            for eid, e in graph.entities.items()
        },
        "relationships": [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation_type": r.relation_type,
                "weight": r.weight,
                "context": r.context,
                "source_video": r.source_video,
            }
            for r in graph.relationships
        ],
        "cross_references": [
            {
                "ref_id": cr.ref_id,
                "source_video": cr.source_video,
                "target_video": cr.target_video,
                "source_context": cr.source_context,
                "target_context": cr.target_context,
                "shared_entities": cr.shared_entities,
                "similarity_score": cr.similarity_score,
                "ref_type": cr.ref_type,
            }
            for cr in graph.cross_references
        ],
        "metadata": graph.graph_metadata,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved knowledge graph to {path}")


def load_knowledge_graph(path: str) -> Optional[KnowledgeGraph]:
    """Load knowledge graph from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities = {}
        for eid, edata in data.get("entities", {}).items():
            entities[eid] = Entity(**edata)

        relationships = [Relationship(**r) for r in data.get("relationships", [])]
        cross_refs = [CrossReference(**cr) for cr in data.get("cross_references", [])]

        return KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            cross_references=cross_refs,
            total_entities=len(entities),
            total_relationships=len(relationships),
            total_cross_references=len(cross_refs),
            graph_metadata=data.get("metadata", {}),
        )

    except FileNotFoundError:
        logger.info(f"No existing knowledge graph at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {e}")
        return None


def query_graph(
    graph: KnowledgeGraph,
    entity_name: Optional[str] = None,
    entity_type: Optional[str] = None,
    video_id: Optional[str] = None,
    max_depth: int = 2,
) -> Dict[str, Any]:
    """
    Query the knowledge graph for connected entities and relationships.

    Args:
        graph: The knowledge graph to query
        entity_name: Name to search for (fuzzy match)
        entity_type: Filter by entity type
        video_id: Filter by source video
        max_depth: Maximum relationship traversal depth

    Returns:
        Dict with matching entities, relationships, and cross-references
    """
    matching_entities: List[Entity] = []

    for entity in graph.entities.values():
        match = True

        if entity_name:
            name_lower = entity_name.lower()
            if (name_lower not in entity.name.lower() and
                not any(name_lower in a.lower() for a in entity.aliases)):
                match = False

        if entity_type and entity.entity_type != entity_type:
            match = False

        if video_id and video_id not in entity.source_videos:
            match = False

        if match:
            matching_entities.append(entity)

    # Find connected relationships
    entity_ids = {e.entity_id for e in matching_entities}
    connected_rels = []
    connected_entities = set()

    # BFS up to max_depth
    current_ids = set(entity_ids)
    for depth in range(max_depth):
        new_ids: Set[str] = set()
        for rel in graph.relationships:
            if rel.source_id in current_ids:
                connected_rels.append(rel)
                new_ids.add(rel.target_id)
            elif rel.target_id in current_ids:
                connected_rels.append(rel)
                new_ids.add(rel.source_id)
        connected_entities.update(new_ids)
        current_ids = new_ids - entity_ids - connected_entities

    # Find related cross-references
    matching_videos = set()
    for e in matching_entities:
        matching_videos.update(e.source_videos)

    related_refs = [
        cr for cr in graph.cross_references
        if cr.source_video in matching_videos or cr.target_video in matching_videos
    ]

    return {
        "entities": matching_entities,
        "relationships": connected_rels[:50],
        "cross_references": related_refs[:20],
        "connected_entity_count": len(connected_entities),
    }


def get_graph_statistics(graph: KnowledgeGraph) -> Dict[str, Any]:
    """Get statistics about the knowledge graph."""
    type_counts = _count_by_type(graph.entities)

    # Most connected entities
    connection_counts: Dict[str, int] = defaultdict(int)
    for rel in graph.relationships:
        connection_counts[rel.source_id] += 1
        connection_counts[rel.target_id] += 1

    top_connected = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_entities = []
    for eid, count in top_connected:
        if eid in graph.entities:
            top_entities.append({
                "name": graph.entities[eid].name,
                "type": graph.entities[eid].entity_type,
                "connections": count,
            })

    # Relationship type distribution
    rel_type_counts: Dict[str, int] = defaultdict(int)
    for rel in graph.relationships:
        rel_type_counts[rel.relation_type] += 1

    # Video coverage
    all_videos: Set[str] = set()
    for entity in graph.entities.values():
        all_videos.update(entity.source_videos)

    return {
        "total_entities": graph.total_entities,
        "total_relationships": graph.total_relationships,
        "total_cross_references": graph.total_cross_references,
        "entity_types": type_counts,
        "relationship_types": dict(rel_type_counts),
        "top_connected_entities": top_entities,
        "videos_covered": len(all_videos),
        "avg_entities_per_video": graph.total_entities / max(len(all_videos), 1),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_entity_id(name: str, entity_type: str) -> str:
    """Generate a deterministic entity ID."""
    key = f"{entity_type}:{name.lower().strip()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _make_ref_id(video1: str, video2: str) -> str:
    """Generate a deterministic cross-reference ID."""
    key = ":".join(sorted([video1, video2]))
    return f"xref-{hashlib.md5(key.encode()).hexdigest()[:10]}"


def _is_common_phrase(text: str) -> bool:
    """Check if a capitalized phrase is a common English phrase (not a name)."""
    common = {
        "The", "This", "That", "There", "These", "Those",
        "What", "When", "Where", "Why", "How",
        "Very", "Also", "Just", "Even", "Still",
    }
    first_word = text.split()[0] if text.split() else ""
    return first_word in common


def _count_by_type(entities: Dict[str, Entity]) -> Dict[str, int]:
    """Count entities by type."""
    counts: Dict[str, int] = defaultdict(int)
    for entity in entities.values():
        counts[entity.entity_type] += 1
    return dict(counts)


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
