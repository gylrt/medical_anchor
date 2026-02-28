import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.config import settings

SPOT_CHECK_TITLE = "Asthma"


def main():
    manifest_path = Path(settings.chroma_dir) / "ingest_manifest.json"
    if not manifest_path.exists():
        print("ERROR: ingest_manifest.json not found. Run ingest.py first.")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print("=== Manifest ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")

    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(settings.collection_name)
    print(f"\n  Total chunks in collection: {collection.count()}")

    # Pull chunks by metadata filter to verify storage is correct
    print(f"\n=== Spot Check: '{SPOT_CHECK_TITLE}' ===")
    results = collection.get(
        where={"topic_title": SPOT_CHECK_TITLE},
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        print(f"  WARNING: no chunks found for '{SPOT_CHECK_TITLE}'")
        return

    print(f"  Chunks found: {len(results['ids'])}")

    meta = results["metadatas"][0]
    doc = results["documents"][0]
    print(f"\n  Sample chunk 0:")
    print(f"  mesh_terms: {json.loads(meta['mesh_terms'])}")
    print(f"  url:        {meta['url']}")
    print(f"  text:       {doc[:200]}...")


if __name__ == "__main__":
    main()