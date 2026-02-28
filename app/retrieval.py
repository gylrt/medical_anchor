import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ner import Entity

CANDIDATE_K = 5
MATCH_THRESHOLD = int(settings.mesh_match_threshold)

# Priority order for match field
_FIELD_PRIORITY = {"title": 0, "synonyms": 1, "see_references": 2, "mesh_terms": 3}


@dataclass
class RetrievalResult:
    entity: Entity
    matched: bool
    passage: Optional[str] = None
    topic_title: Optional[str] = None
    source_url: Optional[str] = None
    distance: Optional[float] = None
    mesh_terms: Optional[str] = None
    synonyms: Optional[str] = None
    see_references: Optional[str] = None


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(settings.collection_name)


def _match_priority(entity_text: str, meta: dict) -> Optional[int]:
    """Return priority of best matching field, or None if no match passes threshold."""
    e = entity_text.lower()
    if fuzz.partial_ratio(e, meta["topic_title"].lower()) >= MATCH_THRESHOLD:
        return _FIELD_PRIORITY["title"]
    for field in ("synonyms", "see_references", "mesh_terms"):
        terms = json.loads(meta.get(field, "[]"))
        if any(fuzz.partial_ratio(e, t.lower()) >= MATCH_THRESHOLD for t in terms):
            return _FIELD_PRIORITY[field]
    return None


def _is_valid_match(entity_text: str, meta: dict) -> bool:
    return _match_priority(entity_text, meta) is not None


def _best_chunk_for_topic(topic_title: str, query_vector: List[float], collection: chromadb.Collection) -> Optional[tuple]:
    """Among all chunks for a topic, return the one closest to the query vector."""
    chunks = collection.get(where={"topic_title": topic_title}, include=["metadatas", "embeddings"])
    if not chunks["ids"]:
        return None
    q = np.array(query_vector)
    best_meta, best_dist = None, float("inf")
    for meta, emb in zip(chunks["metadatas"], chunks["embeddings"]):
        v = np.array(emb)
        dist = 1 - float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
        if dist < best_dist:
            best_dist, best_meta = dist, meta
    return best_meta, best_dist


def retrieve_for_entities(
    entities: List[Entity],
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    max_distance: float = settings.max_distance,
) -> List[RetrievalResult]:
    results = []

    for entity in entities:
        query_vector = embed_model.encode(entity.text).tolist()
        hits = collection.query(
            query_embeddings=[query_vector],
            n_results=CANDIDATE_K,
            include=["metadatas", "distances"],
        )

        # Score each candidate - gate match preferred, distance-only as fallback
        valid, fallback = [], None
        for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
            priority = _match_priority(entity.text, meta)
            if priority is not None:
                valid.append((meta, dist, priority))
            elif dist <= max_distance and fallback is None:
                fallback = (meta, dist)

        if valid:
            # Best topic: field priority > distance
            best_topic = min(valid, key=lambda x: (x[2], x[1]))[0]
            chunk = _best_chunk_for_topic(best_topic["topic_title"], query_vector, collection)
            if chunk is None:
                results.append(RetrievalResult(entity=entity, matched=False))
                continue
            meta, dist = chunk
        elif fallback:
            meta, dist = fallback
        else:
            results.append(RetrievalResult(entity=entity, matched=False))
            continue

        results.append(RetrievalResult(
            entity=entity,
            matched=True,
            passage=meta.get("parent_passage"),
            topic_title=meta.get("topic_title"),
            source_url=meta.get("url"),
            distance=round(dist, 3),
            mesh_terms=meta.get("mesh_terms"),
            synonyms=meta.get("synonyms"),
            see_references=meta.get("see_references"),
        ))

    return results