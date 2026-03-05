import json
from datetime import datetime
from pathlib import Path
import re
import argparse
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.sources.dailymed.chunk import load_dailymed_records_jsonl, build_all_chunks_for_dailymed
from app.config import settings
from app.sources.dailymed.transform import dedup_dailymed_records


def _slug(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "section"


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _primary_drug_name(record: Dict) -> str:
    name_codes = record.get("drug_name_codes", [])
    generic = next((x.get("name") for x in name_codes if x.get("name_type") == "generic" and x.get("name")), None)
    if generic:
        return generic
    first_name = next((x.get("name") for x in name_codes if x.get("name")), None)
    if first_name:
        return first_name
    syns = record.get("synonyms", [])
    if syns:
        return syns[0]
    return record.get("set_id") or record.get("document_id") or "unknown_drug"


def _topic_title(record: Dict) -> str:
    folder = (record.get("folder") or "").strip().lower()
    name = _primary_drug_name(record)
    return f"{name} [{folder}]" if folder else name


def _generic_name(record: Dict) -> str:
    for item in record.get("drug_name_codes", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("name_type", "")).lower() != "generic":
            continue
        name = (item.get("name") or "").strip()
        if name:
            return name
    return ""


def _build_name_index(records: List[Dict]) -> Dict:
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "source": "dailymed",
        "match_order": ["normalized_names", "title", "synonyms"],
        "entries": {
            "normalized_names": {},
            "title": {},
            "synonyms": {},
        },
    }

    for record in records:
        topic_id = record.get("document_id") or record.get("set_id") or ""
        if not topic_id:
            continue
        candidate = {
            "topic_id": topic_id,
            "topic_title": _topic_title(record),
            "generic_name": _generic_name(record),
            "source_url": record.get("source_url", ""),
            "folder": (record.get("folder") or "").strip().lower(),
            "effective_time": record.get("effective_time", ""),
        }

        norm_aliases = set()
        for x in record.get("drug_name_codes", []):
            if not isinstance(x, dict):
                continue
            alias = _normalize_text(x.get("normalized_name") or x.get("name") or "")
            if alias:
                norm_aliases.add(alias)

        title_alias = _normalize_text(candidate["topic_title"])
        syn_aliases = set(_normalize_text(s) for s in record.get("synonyms", []) if _normalize_text(s))

        for alias in norm_aliases:
            payload["entries"]["normalized_names"].setdefault(alias, []).append(candidate)
        if title_alias:
            payload["entries"]["title"].setdefault(title_alias, []).append(candidate)
        for alias in syn_aliases:
            payload["entries"]["synonyms"].setdefault(alias, []).append(candidate)

    return payload


def write_name_index(records: List[Dict]) -> Path:
    out_path = Path(settings.dailymed_name_index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_name_index(records)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Name index written to {out_path}")
    return out_path


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=settings.dailymed_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def write_ingest_manifest(input_jsonl: Path, record_count: int, chunk_count: int, name_index_path: Path):
    manifest = {
        "source_file": input_jsonl.name,
        "ingested_at": datetime.utcnow().isoformat(),
        "record_count": record_count,
        "chunk_count": chunk_count,
        "embed_model": settings.embed_model,
        "collection_name": settings.dailymed_collection_name,
        "name_index_file": str(name_index_path),
    }
    manifest_path = Path(settings.chroma_dir) / "ingest_manifest_dailymed.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


def ingest_dailymed(
    jsonl_path: str,
    dedup: bool = True,
    dedup_similarity_threshold: float = 80.0,
    require_boxed_warning_similarity: bool = True,
):
    input_path = Path(jsonl_path)

    print(f"Loading embedding model: {settings.embed_model}")
    model = SentenceTransformer(settings.embed_model)
    collection = get_collection()

    print(f"Loading DailyMed records from: {input_path}")
    records = load_dailymed_records_jsonl(str(input_path))
    print(f"  -> {len(records)} records loaded")
    if dedup:
        before = len(records)
        records = dedup_dailymed_records(
            records,
            similarity_threshold=dedup_similarity_threshold,
            require_boxed_warning_similarity=require_boxed_warning_similarity,
        )
        print(
            f"  -> {len(records)} records after dedup "
            f"(dropped {before - len(records)}, threshold={dedup_similarity_threshold}, "
            f"boxed_warning_check={require_boxed_warning_similarity})"
        )
    name_index_path = write_name_index(records)

    ids, texts, metadatas = [], [], []
    id_counts = {}
    for chunk_text, meta in build_all_chunks_for_dailymed(records):
        meta = dict(meta)
        meta.pop("parent_passage", None)
        doc_id = meta.get("document_id") or meta.get("set_id") or meta.get("topic_id")
        base_id = (
            f"{doc_id}__"
            f"{meta.get('section_tag', 'other')}__"
            f"{_slug(meta.get('section_name', 'section'))}__"
            f"{meta.get('chunk_index', 0)}"
        )
        seen = id_counts.get(base_id, 0)
        id_counts[base_id] = seen + 1
        chunk_id = base_id if seen == 0 else f"{base_id}__dup{seen}"
        ids.append(chunk_id)
        texts.append(chunk_text)
        metadatas.append(meta)

    print(f"  -> {len(ids)} chunks total")

    total_batches = (len(ids) - 1) // settings.batch_size + 1
    for i in range(0, len(ids), settings.batch_size):
        batch_texts = texts[i : i + settings.batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids[i : i + settings.batch_size],
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=metadatas[i : i + settings.batch_size],
        )
        print(f"  batch {i // settings.batch_size + 1}/{total_batches}")

    write_ingest_manifest(input_path, len(records), len(ids), name_index_path)
    print(f"Done. Collection size: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DailyMed minimal JSONL into Chroma.")
    parser.add_argument(
        "input_jsonl",
        nargs="?",
        default="data/processed/dailymed_minimal.jsonl",
        help="Path to DailyMed minimal JSONL file.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable in-memory dedup before chunking/ingestion.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=80.0,
        help="Indications/boxed-warning similarity threshold used by dedup.",
    )
    parser.add_argument(
        "--ignore-boxed-warning-similarity",
        action="store_true",
        help="Deduplicate using indications similarity only.",
    )
    args = parser.parse_args()

    ingest_dailymed(
        args.input_jsonl,
        dedup=not args.no_dedup,
        dedup_similarity_threshold=args.dedup_threshold,
        require_boxed_warning_similarity=not args.ignore_boxed_warning_similarity,
    )
