"""
Inspect script for DailyMed chunking.

Runs dedup + chunking inspection in one pass:
- loads minimal DailyMed JSONL records
- applies app.dedup_dailymed
- writes deduped JSONL
- builds chunks

Verifies chunk output from deduplicated records:
- total chunk count
- size distribution
- section tag distribution
- spot-check on one drug record
"""

from collections import Counter
import argparse
import json
from pathlib import Path

from app.chunk_dailymed import load_dailymed_records_jsonl, build_all_chunks_for_dailymed
from app.dedup_dailymed import dedup_dailymed_records


DEFAULT_INPUT = Path("data/processed/dailymed_minimal_jan2026_deduped.jsonl")
DEDUP_OUTPUT = Path("data/processed/dailymed_minimal_jan2026_deduped.jsonl")
DEDUP_SIMILARITY_THRESHOLD = 80.0
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
    parser = argparse.ArgumentParser(description="Inspect DailyMed chunking.")
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT),
        help="Input JSONL for chunking (deduped file recommended).",
    )
    parser.add_argument(
        "--build-dedup-from",
        default="",
        help="Optional minimal JSONL to dedup first and write into --input-jsonl path.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEDUP_SIMILARITY_THRESHOLD,
        help="Dedup similarity threshold when using --build-dedup-from.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    dedup_source = Path(args.build_dedup_from) if args.build_dedup_from else None

    if dedup_source:
        if not dedup_source.exists():
            print(f"ERROR: Missing dedup source file: {dedup_source}")
            return
        records = load_dailymed_records_jsonl(str(dedup_source))
        deduped_records = dedup_dailymed_records(records, similarity_threshold=args.dedup_threshold)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("w", encoding="utf-8") as f:
            for rec in deduped_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records_for_chunking = deduped_records
        input_records_count = len(records)
        dedup_records_count = len(deduped_records)
    else:
        if not input_path.exists():
            print(f"ERROR: Missing {input_path}")
            print("Provide --build-dedup-from to generate it, or pass a valid --input-jsonl.")
            return
        records_for_chunking = load_dailymed_records_jsonl(str(input_path))
        input_records_count = len(records_for_chunking)
        dedup_records_count = len(records_for_chunking)

    all_chunks = list(build_all_chunks_for_dailymed(records_for_chunking))

    if not all_chunks:
        print("ERROR: No chunks built from input records.")
        return

    lengths = [len(text) for text, _ in all_chunks]
    tags = Counter(meta.get("section_tag", "unknown") for _, meta in all_chunks)
    folders = Counter(meta.get("folder", "") for _, meta in all_chunks)

    print("=== Summary ===")
    print(f"  Input JSONL:     {input_path}")
    print(f"  Input records:   {input_records_count}")
    print(f"  Dedup records:   {dedup_records_count}")
    print(f"  Dropped records: {input_records_count - dedup_records_count}")
    if dedup_source:
        print(f"  Dedup source:    {dedup_source}")
        print(f"  Dedup threshold: {args.dedup_threshold}")
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

    # Spot-check one drug by primary name
    spot = next((r for r in records_for_chunking if _primary_drug_name(r).lower() == SPOT_CHECK_DRUG.lower()), None)
    if spot is None:
        print(f"\n=== Spot Check ===\n  WARNING: '{SPOT_CHECK_DRUG}' not found in deduped input.")
        return

    spot_id = spot.get("set_id") or spot.get("document_id")
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
