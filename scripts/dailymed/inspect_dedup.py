"""
Inspect script for DailyMed dedup metrics.
"""

from collections import Counter, defaultdict
from pathlib import Path

from app.sources.dailymed.chunk import load_dailymed_records_jsonl
from app.sources.dailymed.transform import dedup_dailymed_records


INPUT_JSONL = Path("data/processed/dailymed_minimal.jsonl")
DEDUP_SIMILARITY_THRESHOLD = 80.0
REQUIRE_BOXED_WARNING_SIMILARITY = False


def _bucket_score(v: float) -> str:
    if v >= 95:
        return "95-100"
    if v >= 90:
        return "90-95"
    if v >= 80:
        return "80-90"
    if v >= 60:
        return "60-80"
    return "<60"


def main():
    if not INPUT_JSONL.exists():
        print(f"ERROR: missing input file: {INPUT_JSONL}")
        return

    raw_records = load_dailymed_records_jsonl(str(INPUT_JSONL))
    records = dedup_dailymed_records(
        raw_records,
        similarity_threshold=DEDUP_SIMILARITY_THRESHOLD,
        require_boxed_warning_similarity=REQUIRE_BOXED_WARNING_SIMILARITY,
    )

    if not records:
        print("ERROR: dedup output is empty.")
        return

    by_folder = Counter((r.get("folder") or "").lower() for r in records)
    winners = [r for r in records if bool(r.get("dedup_is_winner", False))]
    non_winners = [r for r in records if not bool(r.get("dedup_is_winner", False))]

    by_key = defaultdict(list)
    for r in records:
        by_key[r.get("dedup_key", "")].append(r)

    original_estimate = 0
    for r in winners:
        original_estimate += int(r.get("dedup_group_size", 1) or 1)
    dropped_estimate = original_estimate - len(records)

    group_output_sizes = Counter(len(v) for v in by_key.values())
    multi_output_groups = sum(1 for v in by_key.values() if len(v) > 1)

    sim_values = []
    boxed_sim_values = []
    sim_buckets = Counter()
    boxed_sim_buckets = Counter()
    for r in non_winners:
        s = float(r.get("dedup_similarity_to_winner", 0.0) or 0.0)
        sim_values.append(s)
        sim_buckets[_bucket_score(s)] += 1

        bs = float(r.get("dedup_boxed_warning_similarity_to_winner", 100.0) or 100.0)
        boxed_sim_values.append(bs)
        boxed_sim_buckets[_bucket_score(bs)] += 1

    print("=== Deduped Dataset ===")
    print(f"  input_file: {INPUT_JSONL}")
    print(f"  input_records: {len(raw_records)}")
    print(f"  output_records: {len(records)}")
    print(f"  dropped_records: {len(raw_records) - len(records)}")
    print(f"  winners: {len(winners)}")
    print(f"  kept_non_winners: {len(non_winners)}")
    print(f"  estimated_original_records: {original_estimate}")
    print(f"  estimated_dropped_records: {dropped_estimate}")
    print(f"  dedup_threshold: {DEDUP_SIMILARITY_THRESHOLD}")
    print(f"  boxed_warning_check: {REQUIRE_BOXED_WARNING_SIMILARITY}")

    print("\n=== Output Group Sizes (after dedup) ===")
    print(f"  groups_total: {len(by_key)}")
    print(f"  groups_with_multiple_records: {multi_output_groups}")
    for size in sorted(group_output_sizes):
        print(f"  size={size}: {group_output_sizes[size]}")

    print("\n=== Folder Distribution ===")
    for folder, count in by_folder.most_common():
        print(f"  {folder or 'unknown'}: {count}")

    if sim_values:
        print("\n=== Kept Non-Winner Similarity (Indications) ===")
        print(f"  min: {min(sim_values):.1f}")
        print(f"  max: {max(sim_values):.1f}")
        print(f"  avg: {sum(sim_values) / len(sim_values):.1f}")
        for b in ("95-100", "90-95", "80-90", "60-80", "<60"):
            print(f"  {b}: {sim_buckets[b]}")

    if boxed_sim_values:
        print("\n=== Kept Non-Winner Similarity (Boxed Warning) ===")
        print(f"  min: {min(boxed_sim_values):.1f}")
        print(f"  max: {max(boxed_sim_values):.1f}")
        print(f"  avg: {sum(boxed_sim_values) / len(boxed_sim_values):.1f}")
        for b in ("95-100", "90-95", "80-90", "60-80", "<60"):
            print(f"  {b}: {boxed_sim_buckets[b]}")


if __name__ == "__main__":
    main()
