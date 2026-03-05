import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from rapidfuzz import fuzz


FOLDER_PRIORITY = {
    "otc": 0,
    "prescription": 1,
    "homeopathic": 2,
    "animal": 3,
    "other": 4,
}


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_dedup_key(record: Dict) -> str:
    """
    Build stable dedup key from parsed minimal DailyMed record.
    Priority: generic normalized name > first normalized name code > first synonym.
    """
    name_codes = record.get("drug_name_codes", [])
    generic = next(
        (x for x in name_codes if x.get("name_type") == "generic" and x.get("normalized_name")),
        None,
    )
    if generic:
        return generic["normalized_name"]

    first_code = next((x.get("normalized_name") for x in name_codes if x.get("normalized_name")), "")
    if first_code:
        return first_code

    synonyms = record.get("synonyms", [])
    if synonyms:
        return _normalize(synonyms[0])

    return record.get("set_id") or record.get("document_id") or "unknown"


def _effective_time_rank(v: str) -> int:
    v = (v or "").strip()
    if re.fullmatch(r"\d{8}", v):
        return int(v)
    return -1


def _record_rank(record: Dict) -> Tuple[int, int, int, str]:
    folder = (record.get("folder") or "").strip().lower()
    folder_rank = FOLDER_PRIORITY.get(folder, 999)
    # Prefer entries that include boxed warning text.
    has_boxed_warning = 0 if (record.get("boxed_warning") or "").strip() else 1
    # More recent date should win -> negative rank for descending.
    eff_rank = _effective_time_rank(record.get("effective_time", ""))
    set_id = record.get("set_id") or ""
    return (folder_rank, has_boxed_warning, -eff_rank, set_id)


def _indications_similarity(left: Dict, right: Dict) -> float:
    a = (left.get("indications_and_usage") or "").strip()
    b = (right.get("indications_and_usage") or "").strip()
    if not a and not b:
        return 100.0
    if a == b:
        return 100.0
    return float(fuzz.token_sort_ratio(a, b))


def dedup_dailymed_records(
    records: List[Dict],
    similarity_threshold: float = 80.0,
    require_boxed_warning_similarity: bool = True,
) -> List[Dict]:
    """
    Deduplicate minimal DailyMed records.

    Ranking:
    1) folder priority: otc > prescription > homeopathic > animal > other
    2) prefer entries with boxed warning text
    3) latest effective_time
    4) set_id lexical tie-breaker for deterministic output
    5) only drop records that are highly similar to selected winner
       on indications and compatible on boxed warning
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        groups[build_dedup_key(record)].append(record)

    deduped: List[Dict] = []
    for key, group in groups.items():
        ranked = sorted(group, key=_record_rank)
        winner = dict(ranked[0])
        winner["dedup_key"] = key
        winner["dedup_group_size"] = len(group)
        winner["dedup_selected_folder_priority"] = FOLDER_PRIORITY.get(
            (winner.get("folder") or "").strip().lower(),
            999,
        )
        winner["dedup_similarity_to_winner"] = 100.0
        winner["dedup_is_winner"] = True
        deduped.append(winner)

        for candidate in ranked[1:]:
            sim_indications = _indications_similarity(ranked[0], candidate)
            winner_boxed = (ranked[0].get("boxed_warning") or "").strip()
            candidate_boxed = (candidate.get("boxed_warning") or "").strip()

            if winner_boxed and candidate_boxed:
                sim_boxed = float(fuzz.token_sort_ratio(winner_boxed, candidate_boxed))
            else:
                sim_boxed = 100.0

            boxed_ok = (not require_boxed_warning_similarity) or (sim_boxed >= similarity_threshold)
            if sim_indications >= similarity_threshold and boxed_ok:
                continue

            kept = dict(candidate)
            kept["dedup_key"] = key
            kept["dedup_group_size"] = len(group)
            kept["dedup_selected_folder_priority"] = FOLDER_PRIORITY.get(
                (winner.get("folder") or "").strip().lower(),
                999,
            )
            kept["dedup_similarity_to_winner"] = round(sim_indications, 1)
            kept["dedup_boxed_warning_similarity_to_winner"] = round(sim_boxed, 1)
            kept["dedup_is_winner"] = False
            kept["dedup_winner_set_id"] = winner.get("set_id", "")
            deduped.append(kept)

    return deduped


def load_records_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_records_jsonl(records: List[Dict], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dedup_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    similarity_threshold: float = 80.0,
    require_boxed_warning_similarity: bool = True,
) -> Dict[str, int]:
    records = load_records_jsonl(input_jsonl)
    deduped = dedup_dailymed_records(
        records,
        similarity_threshold=similarity_threshold,
        require_boxed_warning_similarity=require_boxed_warning_similarity,
    )
    write_records_jsonl(deduped, output_jsonl)
    return {
        "input_records": len(records),
        "output_records": len(deduped),
        "dropped_records": len(records) - len(deduped),
    }


def main():
    parser = argparse.ArgumentParser(description="Deduplicate DailyMed minimal JSONL records.")
    parser.add_argument(
        "--input-jsonl",
        default="data/processed/dailymed_minimal.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/processed/dailymed_minimal_deduped.jsonl",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--ignore-boxed-warning-similarity",
        action="store_true",
        help="Deduplicate using indications similarity only.",
    )
    args = parser.parse_args()

    stats = dedup_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        similarity_threshold=args.similarity_threshold,
        require_boxed_warning_similarity=not args.ignore_boxed_warning_similarity,
    )
    print("=== DailyMed Dedup ===")
    print(f"input_records: {stats['input_records']}")
    print(f"output_records: {stats['output_records']}")
    print(f"dropped_records: {stats['dropped_records']}")
    print(f"similarity_threshold: {args.similarity_threshold}")
    print(f"output_file: {args.output_jsonl}")


if __name__ == "__main__":
    main()
