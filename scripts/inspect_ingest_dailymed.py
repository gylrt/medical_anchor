import argparse
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.config import settings

DEFAULT_SPOT_CHECK_TITLE = "Fluvoxamine Maleate [prescription]"


def _loads_json_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def main():
    parser = argparse.ArgumentParser(description="Inspect DailyMed ingestion output.")
    parser.add_argument(
        "--spot-title",
        default=DEFAULT_SPOT_CHECK_TITLE,
        help="topic_title value to spot-check in the DailyMed collection.",
    )
    args = parser.parse_args()

    manifest_path = Path(settings.chroma_dir) / "ingest_manifest_dailymed.json"
    if not manifest_path.exists():
        print("ERROR: ingest_manifest_dailymed.json not found. Run app.ingest_dailymed first.")
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
    collection = client.get_collection(settings.dailymed_collection_name)
    print(f"\n  Total chunks in collection: {collection.count()}")

    print(f"\n=== Spot Check: '{args.spot_title}' ===")
    results = collection.get(
        where={"topic_title": args.spot_title},
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        print(f"  WARNING: no chunks found for '{args.spot_title}'")
        print("  Use --spot-title with an existing DailyMed topic_title value.")
        return

    print(f"  Chunks found: {len(results['ids'])}")

    meta = results["metadatas"][0]
    doc = results["documents"][0]
    synonyms = _loads_json_list(meta.get("synonyms"))
    drug_name_codes = _loads_json_list(meta.get("drug_name_codes"))

    primary_name = ""
    generic_names = []
    brand_names = []
    for item in drug_name_codes:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        if not primary_name:
            primary_name = name
        name_type = (item.get("name_type") or "").strip().lower()
        if name_type == "generic" and name not in generic_names:
            generic_names.append(name)
        if name_type == "brand" and name not in brand_names:
            brand_names.append(name)

    print("\n  Sample chunk 0:")
    print(f"  section:    {meta.get('section_name')}")
    print(f"  folder:     {meta.get('folder')}")
    print(f"  source:     {meta.get('source')}")
    print(f"  set_id:     {meta.get('set_id')}")
    print(f"  document_id:{meta.get('document_id')}")
    print(f"  url:        {meta.get('url')}")
    print(f"  primary:    {primary_name or 'n/a'}")
    print(f"  generic:    {generic_names if generic_names else '[]'}")
    print(f"  brand:      {brand_names if brand_names else '[]'}")
    print(f"  synonyms:   {synonyms if synonyms else '[]'}")
    print(f"  text:       {doc[:200]}...")


if __name__ == "__main__":
    main()
