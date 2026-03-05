"""
Inspect script for parse_medlineplus.py

Verifies parser output: topic count, field completeness, a spot-check
on a known topic, and a few full examples with sections and linked mentions.
Run after inspect_download.py to confirm the parsing layer is healthy.
"""

from pathlib import Path
from app.sources.medlineplus.parse import parse_medlineplus_topics

SPOT_CHECK_TITLE = "Asthma"
PREVIEW_COUNT = 3


def main():
    # --- Locate XML ---
    xml_files = sorted(Path("data/raw").glob("mplus_topics_*.xml"))
    if not xml_files:
        print("ERROR: No mplus_topics_*.xml found in data/raw. Run download_medlineplus.py first.")
        return

    xml_path = str(xml_files[-1])
    print(f"Using: {xml_path}\n")

    # --- Parse ---
    topics = parse_medlineplus_topics(xml_path, english_only=True)
    print(f"=== Summary ===")
    print(f"  Total topics parsed: {len(topics)}")

    # Field completeness
    with_url = sum(1 for t in topics if t.url)
    with_summary = sum(1 for t in topics if any(s.name == "Summary" for s in t.sections))
    with_groups = sum(1 for t in topics if t.group_titles)
    with_synonyms = sum(1 for t in topics if t.synonyms)
    with_linked = sum(1 for t in topics if t.linked_mentions)
    with_mesh = sum(1 for t in topics if t.mesh_headings)
    with_related = sum(1 for t in topics if t.related_topics)

    print(f"  With URL:            {with_url}")
    print(f"  With summary:        {with_summary}")
    print(f"  With groups:         {with_groups}")
    print(f"  With synonyms:       {with_synonyms}")
    print(f"  With linked mentions:{with_linked}")
    print(f"  With MeSH headings:  {with_mesh}")
    print(f"  With related topics: {with_related}")

    # --- Spot check on known topic ---
    print(f"\n=== Spot Check: '{SPOT_CHECK_TITLE}' ===")
    topic = next((t for t in topics if t.title.lower() == SPOT_CHECK_TITLE.lower()), None)
    if not topic:
        print(f"  WARNING: '{SPOT_CHECK_TITLE}' not found.")
    else:
        print(f"  ID:              {topic.topic_id}")
        print(f"  URL:             {topic.url}")
        print(f"  Synonyms:        {topic.synonyms[:5]}")
        print(f"  Groups:          {list(zip(topic.group_ids, topic.group_titles))}")
        print(f"  Linked mentions: {topic.linked_mentions[:8]}")
        summary = next((s for s in topic.sections if s.name == "Summary"), None)
        if summary:
            print(f"\n  Summary ({len(summary.text)} chars):")
            print(f"  {summary.text[:400]}...")

        print(f"\n  MeSH headings ({len(topic.mesh_headings)}):")
        for mh in topic.mesh_headings:
            print(f"    [{mh.mesh_id}] {mh.term}")

        print(f"\n  Related topics ({len(topic.related_topics)}):")
        for rt in topic.related_topics:
            print(f"    [{rt.topic_id}] {rt.title} — {rt.url}")

    # --- Preview first N topics ---
    print(f"\n=== First {PREVIEW_COUNT} Topics ===")
    for t in topics[:PREVIEW_COUNT]:
        print("-" * 60)
        print(f"  Title:    {t.title}")
        print(f"  ID:       {t.topic_id}")
        print(f"  URL:      {t.url}")
        print(f"  Groups:   {t.group_titles[:3]}")
        print(f"  Synonyms: {t.synonyms[:3]}")
        print(f"  Sections: {[(s.name, len(s.text)) for s in t.sections]}")
        print(f"  Linked:   {t.linked_mentions[:5]}")


if __name__ == "__main__":
    main()
