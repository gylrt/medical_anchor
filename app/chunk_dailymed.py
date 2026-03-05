import json
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from app.chunking import build_chunks_for_section


def load_dailymed_records_jsonl(jsonl_path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _primary_drug_name(record: Dict) -> str:
    name_codes = record.get("drug_name_codes", [])
    generic = next((x.get("name") for x in name_codes if x.get("name_type") == "generic" and x.get("name")), None)
    if generic:
        return generic
    first_name = next((x.get("name") for x in name_codes if x.get("name")), None)
    if first_name:
        return first_name
    synonyms = record.get("synonyms", [])
    if synonyms:
        return synonyms[0]
    return record.get("set_id") or record.get("document_id") or "unknown_drug"


def _iter_sections(record: Dict) -> Iterable[Tuple[str, str]]:
    sections = record.get("sections", [])
    if isinstance(sections, list):
        emitted = False
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            section_key = (sec.get("section_key") or "").strip().lower()
            if section_key == "boxed_warning":
                continue
            text = (sec.get("text") or "").strip()
            if not text:
                continue
            name = (
                sec.get("title")
                or sec.get("name")
                or (sec.get("section_key") or "").replace("_", " ").title()
                or "Section"
            )
            yield str(name).strip(), text
            emitted = True
        if emitted:
            return

    indications = (record.get("indications_and_usage") or "").strip()
    if indications:
        yield "Indications and Usage", indications


def build_all_chunks_for_dailymed(records: List[Dict]):
    for record in records:
        topic_id = record.get("document_id") or record.get("set_id") or "unknown_topic"
        folder = (record.get("folder") or "").strip().lower()
        source_url = record.get("source_url") or ""
        drug_name = _primary_drug_name(record)

        topic_title = f"{drug_name} [{folder}]" if folder else drug_name

        drug_codes = [x.get("code") for x in record.get("drug_name_codes", []) if x.get("code")]
        see_refs = [record.get("set_id", ""), record.get("document_id", "")] + drug_codes
        see_refs = [x for x in see_refs if x]

        base_metadata = {
            "source": "dailymed",
            "folder": folder,
            "url": source_url,
            "set_id": record.get("set_id", ""),
            "document_id": record.get("document_id", ""),
            "effective_time": record.get("effective_time", ""),
            "synonyms": json.dumps(record.get("synonyms", [])),
            "see_references": json.dumps(see_refs),
            "drug_name_codes": json.dumps(record.get("drug_name_codes", [])),
            "dedup_key": record.get("dedup_key", ""),
            "dedup_group_size": int(record.get("dedup_group_size", 1) or 1),
            "dedup_is_winner": bool(record.get("dedup_is_winner", True)),
            "dedup_similarity_to_winner": float(record.get("dedup_similarity_to_winner", 100.0)),
            # Keep medline-compatible fields for current retrieval code path.
            "group_ids": json.dumps([]),
            "group_titles": json.dumps([]),
            "linked_mentions": json.dumps([]),
            "mesh_terms": json.dumps([]),
            "mesh_ids": json.dumps([]),
            "related_topic_ids": json.dumps([]),
        }

        for section_name, section_text in _iter_sections(record):
            chunks = build_chunks_for_section(
                topic_title=topic_title,
                topic_id=topic_id,
                section_name=section_name,
                section_text=section_text,
                base_metadata=base_metadata,
            )
            for chunk_text, meta in chunks:
                yield chunk_text, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DailyMed chunks from JSONL records.")
    parser.add_argument(
        "--input-jsonl",
        default="data/processed/dailymed_minimal.jsonl",
        help="Path to DailyMed minimal JSONL input.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        print(f"ERROR: missing input file: {input_path}")
    else:
        records = load_dailymed_records_jsonl(str(input_path))
        chunks = list(build_all_chunks_for_dailymed(records))
        print(f"input={input_path}")
        print(f"records={len(records)} chunks={len(chunks)}")
