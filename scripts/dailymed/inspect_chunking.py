"""
Inspect script for DailyMed chunking.

Runs one-shot: minimal JSONL -> in-memory dedup -> chunk build.
"""

from collections import Counter
from pathlib import Path

from app.sources.dailymed.chunk import load_dailymed_records_jsonl, build_all_chunks_for_dailymed
from app.sources.dailymed.transform import dedup_dailymed_records


INPUT_JSONL = Path("data/processed/dailymed_minimal.jsonl")
DEDUP_SIMILARITY_THRESHOLD = 80.0
REQUIRE_BOXED_WARNING_SIMILARITY = False
SPOT_CHECK_DRUG = "Fluvoxamine Maleate"


def _primary_drug_name(record: dict) -> str:
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


def main():
    if not INPUT_JSONL.exists():
        print(f"ERROR: Missing {INPUT_JSONL}")
        return

    input_records = load_dailymed_records_jsonl(str(INPUT_JSONL))
    records_for_chunking = dedup_dailymed_records(
        input_records,
        similarity_threshold=DEDUP_SIMILARITY_THRESHOLD,
        require_boxed_warning_similarity=REQUIRE_BOXED_WARNING_SIMILARITY,
    )

    all_chunks = list(build_all_chunks_for_dailymed(records_for_chunking))

    if not all_chunks:
        print("ERROR: No chunks built from input records.")
        return

    lengths = [len(text) for text, _ in all_chunks]
    tags = Counter(meta.get("section_tag", "unknown") for _, meta in all_chunks)
    folders = Counter(meta.get("folder", "") for _, meta in all_chunks)

    print("=== Summary ===")
    print(f"  Input JSONL:     {INPUT_JSONL}")
    print(f"  Input records:   {len(input_records)}")
    print(f"  Dedup records:   {len(records_for_chunking)}")
    print(f"  Dropped records: {len(input_records) - len(records_for_chunking)}")
    print(f"  Dedup threshold: {DEDUP_SIMILARITY_THRESHOLD}")
    print(f"  Boxed check:     {REQUIRE_BOXED_WARNING_SIMILARITY}")
    print(f"  Total chunks:    {len(all_chunks)}")
    print(f"  Avg chars:       {sum(lengths) // len(lengths)}")
    print(f"  Min chars:       {min(lengths)}")
    print(f"  Max chars:       {max(lengths)}")

    print("\n=== Section Tag Distribution ===")
    for tag, count in tags.most_common():
        print(f"  {tag}: {count}")

    print("\n=== Folder Distribution (chunk metadata) ===")
    for folder, count in folders.most_common():
        print(f"  {folder or 'unknown'}: {count}")

    spot = next((r for r in records_for_chunking if _primary_drug_name(r).lower() == SPOT_CHECK_DRUG.lower()), None)
    if spot is None:
        print(f"\n=== Spot Check ===\n  WARNING: '{SPOT_CHECK_DRUG}' not found in deduped input.")
        return

    spot_id = spot.get("document_id") or spot.get("set_id")
    spot_chunks = [(t, m) for t, m in all_chunks if m.get("topic_id") == spot_id]

    print(f"\n=== Spot Check: '{SPOT_CHECK_DRUG}' ===")
    print(f"  topic_id: {spot_id}")
    print(f"  chunks:   {len(spot_chunks)}")

    for text, meta in spot_chunks[:4]:
        print(f"\n  --- Chunk {meta.get('chunk_index')} ({len(text)} chars) ---")
        print(f"  section:   {meta.get('section_name')} ({meta.get('section_tag')})")
        print(f"  source:    {meta.get('source')}")
        print(f"  folder:    {meta.get('folder')}")
        print(f"  url:       {meta.get('url')}")
        print(f"  preview:   {text[:280]}...")


if __name__ == "__main__":
    main()
