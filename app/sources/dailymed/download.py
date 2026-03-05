import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import requests
import re

from app.sources.dailymed.parse import parse_dailymed_labels_from_xml_bytes


DATA_DIR = Path("data/raw")
TMP_DIR = DATA_DIR / "tmp"
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = DATA_DIR / "dailymed_manifest.json"

MONTHLY_URL_TEMPLATE = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_monthly_update_{month}.zip"
FULL_RX_PART_TEMPLATE = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part{part}.zip"
FULL_OTC_PART_TEMPLATE = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_otc_part{part}.zip"
FULL_HOMEOPATHIC_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_homeopathic.zip"
FULL_ANIMAL_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_animal.zip"
FULL_OTHER_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_remainder.zip"

FOLDERS = ("otc", "prescription", "homeopathic", "animal", "other")
MONTHLY_LISTING_URL = "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"


def _download_stream_with_sha256(url: str, output_path: Path, timeout: int = 60) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total = 0
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                h.update(chunk)
                total += len(chunk)
    return {"sha256": h.hexdigest(), "size_bytes": total}


def _append_manifest(entry: Dict):
    payload = {"entries": []}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        if "entries" not in payload:
            payload = {"entries": [payload]}
    payload["entries"].append(entry)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_manifest_entries() -> List[Dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "entries" in payload and isinstance(payload["entries"], list):
        return payload["entries"]
    if isinstance(payload, dict):
        return [payload]
    return []


def _iter_nested_zip_entries(outer: zipfile.ZipFile, folders: Iterable[str]):
    prefixes = tuple(f"{f}/" for f in folders)
    for info in outer.infolist():
        if info.is_dir():
            continue
        if not info.filename.endswith(".zip"):
            continue
        if not info.filename.startswith(prefixes):
            continue
        yield info


def _join_section_text(label, section_key: str) -> str:
    parts = [s.text.strip() for s in label.sections if s.section_key == section_key and s.text.strip()]
    return "\n\n".join(parts).strip()


def _label_to_record(label, folder: str, nested_zip_path: str) -> Dict:
    sections = [
        {
            "section_key": s.section_key,
            "code": s.code,
            "title": (s.title or s.section_key.replace("_", " ").title()),
            "text": s.text,
        }
        for s in label.sections
        if s.text
    ]
    return {
        "folder": folder,
        "nested_zip_path": nested_zip_path,
        "set_id": label.set_id,
        "document_id": label.document_id,
        "version": label.version,
        "effective_time": label.effective_time,
        "source_url": label.source_url,
        "drug_name_codes": [
            {
                "name": n.name,
                "code": n.code,
                "name_type": n.name_type,
                "normalized_name": n.normalized_name,
            }
            for n in label.drug_name_codes
        ],
        "synonyms": list(label.synonyms),
        "sections": sections,
        "indications_and_usage": _join_section_text(label, "indications_and_usage"),
        "boxed_warning": _join_section_text(label, "boxed_warning"),
    }


def process_outer_zip_to_jsonl(zip_path: Path, output_jsonl: Path, max_boxed_warning_chars: int = 2500) -> Dict:
    stats = {
        "processed_nested": 0,
        "written_records": 0,
        "skipped_missing_indications": 0,
        "missing_xml": 0,
        "parse_errors": 0,
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as outer, open(output_jsonl, "a", encoding="utf-8") as out:
        for entry in _iter_nested_zip_entries(outer, FOLDERS):
            stats["processed_nested"] += 1
            folder = entry.filename.split("/", 1)[0]
            try:
                nested_bytes = outer.read(entry)
                with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as inner:
                    xml_names = [n for n in inner.namelist() if n.lower().endswith(".xml")]
                    if not xml_names:
                        stats["missing_xml"] += 1
                        continue
                    xml_bytes = inner.read(xml_names[0])
            except Exception:
                stats["parse_errors"] += 1
                continue

            try:
                labels = parse_dailymed_labels_from_xml_bytes(
                    xml_bytes, max_boxed_warning_chars=max_boxed_warning_chars
                )
                if not labels:
                    stats["parse_errors"] += 1
                    continue
                rec = _label_to_record(labels[0], folder=folder, nested_zip_path=entry.filename)
            except Exception:
                stats["parse_errors"] += 1
                continue

            if not (rec.get("indications_and_usage") or "").strip():
                stats["skipped_missing_indications"] += 1
                continue

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["written_records"] += 1

    return stats


def _collect_part_urls(template: str, max_parts: int) -> List[str]:
    urls = []
    for part in range(1, max_parts + 1):
        url = template.format(part=part)
        r = requests.head(url, timeout=15)
        if r.status_code == 200:
            urls.append(url)
        elif r.status_code == 404:
            break
        else:
            break
    return urls


def _resolve_urls(mode: str, month: str, explicit_url: str, max_parts: int) -> List[str]:
    if mode == "url":
        if not explicit_url:
            raise ValueError("--url is required when mode=url")
        return [explicit_url]
    if mode == "monthly":
        return [MONTHLY_URL_TEMPLATE.format(month=month)]
    if mode == "full-rx":
        return _collect_part_urls(FULL_RX_PART_TEMPLATE, max_parts)
    if mode == "full-otc":
        return _collect_part_urls(FULL_OTC_PART_TEMPLATE, max_parts)
    if mode == "full-homeopathic":
        return [FULL_HOMEOPATHIC_URL]
    if mode == "full-animal":
        return [FULL_ANIMAL_URL]
    if mode == "full-other":
        return [FULL_OTHER_URL]
    if mode == "full-human":
        return _collect_part_urls(FULL_RX_PART_TEMPLATE, max_parts) + _collect_part_urls(FULL_OTC_PART_TEMPLATE, max_parts)
    if mode == "full-all":
        return (
            _collect_part_urls(FULL_RX_PART_TEMPLATE, max_parts)
            + _collect_part_urls(FULL_OTC_PART_TEMPLATE, max_parts)
            + [FULL_HOMEOPATHIC_URL, FULL_ANIMAL_URL, FULL_OTHER_URL]
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _latest_monthly_url() -> str:
    r = requests.get(MONTHLY_LISTING_URL, timeout=30)
    r.raise_for_status()
    html = r.text
    urls = re.findall(
        r"https://dailymed-data\.nlm\.nih\.gov/public-release-files/dm_spl_monthly_update_[a-z]{3}\d{4}\.zip",
        html,
        flags=re.IGNORECASE,
    )
    if not urls:
        raise RuntimeError("Could not find monthly update URL on DailyMed listing page.")
    # Sort by month token in URL is not strictly safe lexicographically, rely on appearance order first.
    # Keep unique while preserving order, then take first match.
    seen = set()
    ordered = []
    for u in urls:
        lu = u.lower()
        if lu in seen:
            continue
        seen.add(lu)
        ordered.append(u)
    return ordered[0]


def _resolve_auto_mode() -> List[str]:
    entries = _load_manifest_entries()
    has_full_baseline = any((e.get("mode") == "full-all") for e in entries)
    if not has_full_baseline:
        return _resolve_urls("full-all", "", "", max_parts=30)
    return [_latest_monthly_url()]


def main():
    parser = argparse.ArgumentParser(description="Download and process DailyMed release ZIPs.")
    parser.add_argument(
        "--mode",
        default="monthly",
        choices=[
            "auto",
            "monthly",
            "full-rx",
            "full-otc",
            "full-homeopathic",
            "full-animal",
            "full-other",
            "full-human",
            "full-all",
            "url",
        ],
    )
    parser.add_argument("--month", default=datetime.utcnow().strftime("%b%Y").lower(), help="e.g. feb2026")
    parser.add_argument("--url", default="", help="Explicit URL when mode=url")
    parser.add_argument("--max-parts", type=int, default=30)
    parser.add_argument("--cleanup-zip", action="store_true")
    parser.add_argument("--max-boxed-warning-chars", type=int, default=2500)
    parser.add_argument(
        "--output-jsonl",
        default=str(PROCESSED_DIR / "dailymed_minimal.jsonl"),
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    if args.mode == "auto":
        urls = _resolve_auto_mode()
        resolved_mode = "auto"
    else:
        urls = _resolve_urls(args.mode, args.month, args.url, args.max_parts)
        resolved_mode = args.mode
    if not urls:
        raise RuntimeError("No URLs resolved for selected mode.")

    print(f"Mode: {resolved_mode}")
    print(f"Archives: {len(urls)}")
    for i, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        zip_path = TMP_DIR / filename
        print(f"\n[{i}/{len(urls)}] Downloading {url}")
        dl = _download_stream_with_sha256(url, zip_path)

        stats = process_outer_zip_to_jsonl(
            zip_path=zip_path,
            output_jsonl=output_jsonl,
            max_boxed_warning_chars=args.max_boxed_warning_chars,
        )
        print(
            "processed_nested={processed_nested} written={written_records} "
            "skipped_missing_indications={skipped_missing_indications} "
            "missing_xml={missing_xml} parse_errors={parse_errors}".format(**stats)
        )

        _append_manifest(
            {
                "source": "DailyMed SPL",
                "mode": resolved_mode,
                "url": url,
                "file": zip_path.name,
                "downloaded_at": datetime.utcnow().isoformat(),
                "sha256": dl["sha256"],
                "size_bytes": dl["size_bytes"],
                "output_jsonl": str(output_jsonl),
                "process_stats": stats,
            }
        )
        print(f"Manifest updated: {MANIFEST_PATH}")

        if args.cleanup_zip:
            zip_path.unlink(missing_ok=True)
            print(f"Deleted: {zip_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
