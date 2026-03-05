import json
from datetime import datetime
from pathlib import Path
import re
import argparse

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


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=settings.dailymed_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def write_ingest_manifest(input_jsonl: Path, record_count: int, chunk_count: int):
    manifest = {
        "source_file": input_jsonl.name,
        "ingested_at": datetime.utcnow().isoformat(),
        "record_count": record_count,
        "chunk_count": chunk_count,
        "embed_model": settings.embed_model,
        "collection_name": settings.dailymed_collection_name,
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

    write_ingest_manifest(input_path, len(records), len(ids))
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
