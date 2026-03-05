import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings
from rapidfuzz import fuzz
from rapidfuzz import process
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ner import Entity

CANDIDATE_K = 5
MATCH_THRESHOLD = int(settings.mesh_match_threshold)

# Priority order for match field
_FIELD_PRIORITY = {"title": 0, "synonyms": 1, "see_references": 2, "mesh_terms": 3}
_DAILYMED_FIELD_PRIORITY = {"normalized_names": 0, "title": 1, "synonyms": 2}

_TREATMENT_LABELS = {"treatment"}
_MEDLINE_LABELS = {"problem", "test"}
_DAILYMED_INDEX_CACHE: Dict[int, Dict[str, dict]] = {}
_DAILYMED_ALIAS_CACHE: Dict[int, Dict[str, dict]] = {}
_DAILYMED_MATCH_CACHE: Dict[Tuple[int, str], Optional[tuple]] = {}
_DAILYMED_INDEX_PAGE_SIZE = 5000
_DAILYMED_NAME_INDEX_CACHE: Optional[dict] = None
_DAILYMED_FOLDER_PRIORITY = {
    "otc": 0,
    "prescription": 1,
    "homeopathic": 2,
    "animal": 3,
    "other": 4,
}


@dataclass
class RetrievalResult:
    entity: Entity
    matched: bool
    passage: Optional[str] = None
    topic_title: Optional[str] = None
    source_url: Optional[str] = None
    generic_name: Optional[str] = None
    distance: Optional[float] = None
    mesh_terms: Optional[str] = None
    synonyms: Optional[str] = None
    see_references: Optional[str] = None


def load_collections() -> Tuple[chromadb.Collection, chromadb.Collection]:
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    medline = client.get_collection(settings.collection_name)
    dailymed = client.get_collection(settings.dailymed_collection_name)
    return medline, dailymed


def load_collection() -> chromadb.Collection:
    # Backward-compatible helper for scripts that only target MedlinePlus.
    medline, _ = load_collections()
    return medline


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


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


def _effective_time_rank(v: str) -> int:
    v = (v or "").strip()
    return int(v) if len(v) == 8 and v.isdigit() else -1


def _dailymed_normalized_names(meta: dict) -> List[str]:
    names = []
    for item in json.loads(meta.get("drug_name_codes", "[]")):
        if not isinstance(item, dict):
            continue
        n = _normalize_text(item.get("normalized_name") or item.get("name") or "")
        if n and n not in names:
            names.append(n)
    for s in json.loads(meta.get("synonyms", "[]")):
        n = _normalize_text(s)
        if n and n not in names:
            names.append(n)
    return names


def _dailymed_generic_name(meta: dict) -> Optional[str]:
    for item in json.loads(meta.get("drug_name_codes", "[]")):
        if not isinstance(item, dict):
            continue
        if str(item.get("name_type", "")).lower() != "generic":
            continue
        name = (item.get("name") or "").strip()
        if name:
            return name
    return None


def _load_dailymed_name_index() -> dict:
    global _DAILYMED_NAME_INDEX_CACHE
    if _DAILYMED_NAME_INDEX_CACHE is not None:
        return _DAILYMED_NAME_INDEX_CACHE
    path = Path(settings.dailymed_name_index_path)
    if not path.exists():
        _DAILYMED_NAME_INDEX_CACHE = {"entries": {"normalized_names": {}, "title": {}, "synonyms": {}}}
        return _DAILYMED_NAME_INDEX_CACHE
    with open(path, encoding="utf-8") as f:
        _DAILYMED_NAME_INDEX_CACHE = json.load(f)
    if "entries" not in _DAILYMED_NAME_INDEX_CACHE:
        _DAILYMED_NAME_INDEX_CACHE = {"entries": {"normalized_names": {}, "title": {}, "synonyms": {}}}
    return _DAILYMED_NAME_INDEX_CACHE


def _dailymed_candidate_key(priority: int, score: float, candidate: dict):
    folder = str(candidate.get("folder", "")).lower()
    folder_rank = _DAILYMED_FOLDER_PRIORITY.get(folder, 999)
    eff_rank = _effective_time_rank(str(candidate.get("effective_time", "")))
    topic_id = str(candidate.get("topic_id", ""))
    return (priority, -score, folder_rank, -eff_rank, topic_id)


def _best_chunk_for_topic_id(topic_id: str, collection: chromadb.Collection):
    chunks = collection.get(where={"topic_id": topic_id}, include=["metadatas", "documents"])
    if not chunks.get("ids"):
        return None
    best = None
    for meta, doc in zip(chunks.get("metadatas", []), chunks.get("documents", [])):
        section_tag = str(meta.get("section_tag", ""))
        chunk_index = int(meta.get("chunk_index", 0) or 0)
        key = (0 if section_tag == "indications" else 1, chunk_index)
        if best is None or key < best[0]:
            best = (key, meta, doc)
    if best is None:
        return None
    return best[1], best[2]


def _best_chunk_for_topic(topic_title: str, query_vector: List[float], collection: chromadb.Collection) -> Optional[tuple]:
    """Among all chunks for a topic, return the one closest to the query vector."""
    chunks = collection.get(where={"topic_title": topic_title}, include=["metadatas", "embeddings", "documents"])
    if not chunks["ids"]:
        return None
    q = np.array(query_vector)
    best_meta, best_dist, best_doc = None, float("inf"), None
    for meta, emb, doc in zip(chunks["metadatas"], chunks["embeddings"], chunks["documents"]):
        v = np.array(emb)
        dist = 1 - float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
        if dist < best_dist:
            best_dist, best_meta, best_doc = dist, meta, doc
    return best_meta, best_dist, best_doc


def _select_target(entity: Entity, medline_collection: chromadb.Collection, dailymed_collection: chromadb.Collection):
    label = (entity.label or "").strip().lower()
    if label in _TREATMENT_LABELS:
        return dailymed_collection, "dailymed"
    if label in _MEDLINE_LABELS:
        return medline_collection, "medlineplus"
    # Safe default for unknown labels.
    return medline_collection, "medlineplus"


def _get_dailymed_topic_index(collection: chromadb.Collection) -> Dict[str, dict]:
    cache_key = id(collection)
    if cache_key in _DAILYMED_INDEX_CACHE:
        return _DAILYMED_INDEX_CACHE[cache_key]

    topics: Dict[str, dict] = {}
    offset = 0
    while True:
        res = collection.get(
            include=["metadatas", "documents"],
            limit=_DAILYMED_INDEX_PAGE_SIZE,
            offset=offset,
        )
        metadatas = res.get("metadatas", [])
        documents = res.get("documents", [])
        if not metadatas:
            break

        for meta, doc in zip(metadatas, documents):
            if not meta:
                continue
            topic_id = meta.get("topic_id")
            if not topic_id:
                continue

            row = topics.get(topic_id)
            section_tag = meta.get("section_tag", "")
            current_is_indications = bool(row and row.get("section_tag") == "indications")
            new_is_indications = section_tag == "indications"

            if row is not None and current_is_indications and not new_is_indications:
                continue

            row = {
                "meta": meta,
                "doc": doc,
                "section_tag": section_tag,
                "title_norm": _normalize_text(meta.get("topic_title", "")),
                "names_norm": _dailymed_normalized_names(meta),
                "synonyms_norm": [_normalize_text(s) for s in json.loads(meta.get("synonyms", "[]")) if s],
                "token_set": set(),
            }
            row["token_set"] = set(
                " ".join([row["title_norm"]] + row["names_norm"] + row["synonyms_norm"]).split()
            )
            topics[topic_id] = row

        offset += _DAILYMED_INDEX_PAGE_SIZE

    _DAILYMED_INDEX_CACHE[cache_key] = topics
    return topics


def _get_dailymed_alias_cache(collection: chromadb.Collection) -> Dict[str, dict]:
    cache_key = id(collection)
    if cache_key in _DAILYMED_ALIAS_CACHE:
        return _DAILYMED_ALIAS_CACHE[cache_key]

    topics = _get_dailymed_topic_index(collection)
    normalized_map: Dict[str, tuple] = {}
    title_map: Dict[str, tuple] = {}
    synonyms_map: Dict[str, tuple] = {}
    for row in topics.values():
        meta = row["meta"]
        folder_rank = _DAILYMED_FOLDER_PRIORITY.get(str(meta.get("folder", "")).lower(), 999)
        eff_rank = _effective_time_rank(str(meta.get("effective_time", "")))
        topic_id = str(meta.get("topic_id", ""))
        key = (folder_rank, -eff_rank, topic_id)

        for alias in [a for a in row["names_norm"] if a]:
            existing = normalized_map.get(alias)
            if existing is None or key < existing[0]:
                normalized_map[alias] = (key, meta, row["doc"])

        if row["title_norm"]:
            existing = title_map.get(row["title_norm"])
            if existing is None or key < existing[0]:
                title_map[row["title_norm"]] = (key, meta, row["doc"])

        for alias in [a for a in row["synonyms_norm"] if a]:
            existing = synonyms_map.get(alias)
            if existing is None or key < existing[0]:
                synonyms_map[alias] = (key, meta, row["doc"])

    payload = {
        "normalized": normalized_map,
        "title": title_map,
        "synonyms": synonyms_map,
        "normalized_keys": list(normalized_map.keys()),
        "title_keys": list(title_map.keys()),
        "synonyms_keys": list(synonyms_map.keys()),
    }
    _DAILYMED_ALIAS_CACHE[cache_key] = payload
    return payload


def _best_dailymed_lexical_match(entity_text: str, collection: chromadb.Collection):
    e_norm = _normalize_text(entity_text)
    if not e_norm:
        return None
    cache_key = (id(collection), e_norm)
    if cache_key in _DAILYMED_MATCH_CACHE:
        return _DAILYMED_MATCH_CACHE[cache_key]

    index = _load_dailymed_name_index()
    entries = index.get("entries", {})

    # Exact then fuzzy in strict order: normalized_names > title > synonyms
    for field, priority in (
        ("normalized_names", _DAILYMED_FIELD_PRIORITY["normalized_names"]),
        ("title", _DAILYMED_FIELD_PRIORITY["title"]),
        ("synonyms", _DAILYMED_FIELD_PRIORITY["synonyms"]),
    ):
        field_map = entries.get(field, {}) or {}
        exact_candidates = field_map.get(e_norm, [])
        if exact_candidates:
            best = min(exact_candidates, key=lambda c: _dailymed_candidate_key(priority, 100.0, c))
            _DAILYMED_MATCH_CACHE[cache_key] = best
            return best

        keys = list(field_map.keys())
        if not keys:
            continue
        match = process.extractOne(
            e_norm,
            keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=MATCH_THRESHOLD,
        )
        if match is None:
            continue
        alias = match[0]
        score = float(match[1])
        cand_list = field_map.get(alias, [])
        if not cand_list:
            continue
        best = min(cand_list, key=lambda c: _dailymed_candidate_key(priority, score, c))
        _DAILYMED_MATCH_CACHE[cache_key] = best
        return best

    _DAILYMED_MATCH_CACHE[cache_key] = None
    return None


def retrieve_for_entities(
    entities: List[Entity],
    medline_collection: chromadb.Collection,
    dailymed_collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    max_distance: float = settings.max_distance,
) -> List[RetrievalResult]:
    results = []

    for entity in entities:
        collection, source = _select_target(entity, medline_collection, dailymed_collection)

        if source == "dailymed":
            match = _best_dailymed_lexical_match(entity.text, collection)
            if match is None:
                results.append(RetrievalResult(entity=entity, matched=False))
                continue
            topic_id = match.get("topic_id", "")
            best = _best_chunk_for_topic_id(topic_id, collection)
            if best is None:
                results.append(RetrievalResult(entity=entity, matched=False))
                continue
            meta, doc = best
            dist = None
        else:
            query_vector = embed_model.encode(entity.text).tolist()
            hits = collection.query(
                query_embeddings=[query_vector],
                n_results=CANDIDATE_K,
                include=["metadatas", "distances", "documents"],
            )

            # Score each candidate - gate match preferred, distance-only as fallback
            valid, fallback = [], None
            for meta, dist, doc in zip(hits["metadatas"][0], hits["distances"][0], hits["documents"][0]):
                priority = _match_priority(entity.text, meta)
                if priority is not None:
                    valid.append((meta, dist, priority, doc))
                elif dist <= max_distance and fallback is None:
                    fallback = (meta, dist, doc)

            if valid:
                # Best topic: field priority > distance
                best_topic = min(valid, key=lambda x: (x[2], x[1]))[0]
                chunk = _best_chunk_for_topic(best_topic["topic_title"], query_vector, collection)
                if chunk is None:
                    results.append(RetrievalResult(entity=entity, matched=False))
                    continue
                meta, dist, doc = chunk
            elif fallback:
                meta, dist, doc = fallback
            else:
                results.append(RetrievalResult(entity=entity, matched=False))
                continue

        results.append(RetrievalResult(
            entity=entity,
            matched=True,
            passage=meta.get("parent_passage") or doc,
            topic_title=meta.get("topic_title"),
            source_url=meta.get("url"),
            generic_name=(
                (match.get("generic_name") or _dailymed_generic_name(meta))
                if source == "dailymed"
                else None
            ),
            distance=round(dist, 3) if isinstance(dist, float) else None,
            mesh_terms=meta.get("mesh_terms"),
            synonyms=meta.get("synonyms"),
            see_references=meta.get("see_references"),
        ))

    return results
