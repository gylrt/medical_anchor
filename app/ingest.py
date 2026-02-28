import os
import json
from pathlib import Path
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.parse_medlineplus import parse_medlineplus_topics
from app.chunking import build_chunks_for_section

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
COLLECTION_NAME = "medlineplus_v1"
BATCH_SIZE = 256


def get_collection(chroma_dir: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def write_ingest_manifest(xml_path: Path, chunk_count: int, chroma_dir: str):
    manifest = {
        "source_file": xml_path.name,
        "ingested_at": datetime.utcnow().isoformat(),
        "chunk_count": chunk_count,
        "embed_model": EMBED_MODEL,
    }
    manifest_path = Path(chroma_dir) / "ingest_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


def build_all_chunks(topics):
    for topic in topics:
        # Chroma only accepts primitives — lists are JSON-encoded strings
        base_metadata = {
            "url": topic.url or "",
            "group_ids": json.dumps(topic.group_ids),
            "group_titles": json.dumps(topic.group_titles),
            "linked_mentions": json.dumps(topic.linked_mentions),
            "mesh_terms": json.dumps([mh.term for mh in topic.mesh_headings]),
            "mesh_ids": json.dumps([mh.mesh_id for mh in topic.mesh_headings]),
            "related_topic_ids": json.dumps([rt.topic_id for rt in topic.related_topics]),
        }
        for section in topic.sections:
            chunks = build_chunks_for_section(
                topic_title=topic.title,
                topic_id=topic.topic_id,
                section_name=section.name,
                section_text=section.text,
                base_metadata=base_metadata,
            )
            for chunk_text, meta in chunks:
                yield chunk_text, meta


def ingest(xml_path: str, chroma_dir: str = CHROMA_DIR):
    xml_path = Path(xml_path)

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    collection = get_collection(chroma_dir)

    print(f"Parsing topics from: {xml_path}")
    topics = parse_medlineplus_topics(str(xml_path), english_only=True)
    print(f"  → {len(topics)} topics parsed")

    ids, texts, metadatas = [], [], []
    for chunk_text, meta in build_all_chunks(topics):
        chunk_id = f"{meta['topic_id']}__{meta['section_tag']}__{meta['chunk_index']}"
        ids.append(chunk_id)
        texts.append(chunk_text)
        metadatas.append(meta)

    print(f"  → {len(ids)} chunks total")

    total_batches = (len(ids) - 1) // BATCH_SIZE + 1
    for i in range(0, len(ids), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids[i : i + BATCH_SIZE],
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=metadatas[i : i + BATCH_SIZE],
        )
        print(f"  batch {i // BATCH_SIZE + 1}/{total_batches}")

    write_ingest_manifest(xml_path, len(ids), chroma_dir)
    print(f"Done. Collection size: {collection.count()}")


if __name__ == "__main__":
    import sys
    xml_path = sys.argv[1] if len(sys.argv) > 1 else str(
        sorted(Path("data/raw").glob("mplus_topics_*.xml"))[-1]
    )
    ingest(xml_path)