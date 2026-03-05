"""
Inspect script for minimal DailyMed parsing.
"""

from pathlib import Path

from app.parse_dailymed import parse_dailymed_labels

DEFAULT_XML = "658eeffd-e919-4d41-8814-e55273a471e9.xml"
MAX_BOXED_WARNING_CHARS = 2500


def main():
    xml_path = Path("data/raw") / DEFAULT_XML
    if not xml_path.exists():
        xml_candidates = sorted(Path("data/raw").glob("*.xml"))
        if not xml_candidates:
            print("ERROR: No XML files found in data/raw.")
            return
        xml_path = xml_candidates[-1]

    labels = parse_dailymed_labels(str(xml_path), max_boxed_warning_chars=MAX_BOXED_WARNING_CHARS)
    if not labels:
        print(f"ERROR: Parser returned no labels for {xml_path}")
        return

    label = labels[0]
    print(f"Using: {xml_path}\n")
    print("=== Minimal Label Metadata ===")
    print(f"document_id:        {label.document_id}")
    print(f"set_id:             {label.set_id}")
    print(f"version:            {label.version}")
    print(f"effective_time:     {label.effective_time}")
    print(f"source_url:         {label.source_url}")
    print(f"drug_name_codes:    {len(label.drug_name_codes)}")
    print(f"synonyms:           {len(label.synonyms)}")
    print(f"sections:           {len(label.sections)}")
    print(f"total_text_chars:   {sum(len(s.text) for s in label.sections)}")

    print("\n=== Drug Name + Code ===")
    for item in label.drug_name_codes:
        print(f"- [{item.name_type}] code={item.code} name={item.name} norm={item.normalized_name}")

    print("\n=== Synonyms ===")
    for s in label.synonyms:
        print(f"- {s}")

    print("\n=== Sections By Priority ===")
    priority = [
        "boxed_warning",
        "indications_and_usage",
    ]
    for key in priority:
        hits = [s for s in label.sections if s.section_key == key]
        chars = sum(len(s.text) for s in hits)
        print(f"- {key}: {len(hits)} sections, {chars} chars")
        for sec in hits[:3]:
            title = sec.title or "Untitled"
            snippet = sec.text[:220].replace("\n", " ")
            print(f"  [{sec.code}] {title}")
            print(f"  {snippet}...")


if __name__ == "__main__":
    main()
