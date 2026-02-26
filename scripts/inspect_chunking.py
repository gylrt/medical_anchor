"""
Inspect script for chunking.py

Verifies chunk output: total chunk count, size distribution, metadata
completeness, and a full preview of chunks for a known topic.
Run after inspect_parser.py to confirm the chunking layer is healthy.
"""

from pathlib import Path
from app.parse_medlineplus import parse_medlineplus_topics
from app.chunking import build_chunks_for_section

SPOT_CHECK_TITLE = "Asthma"


def main():
    # --- Locate XML ---
    xml_files = sorted(Path("data/raw").glob("mplus_topics_*.xml"))
    if not xml_files:
        print("ERROR: No mplus_topics_*.xml found in data/raw. Run download_medlineplus.py first.")
        return

    xml_path = str(xml_files[-1])
    print(f"Using: {xml_path}\n")

    topics = parse_medlineplus_topics(xml_path, english_only=True)

    # --- Build all chunks ---
    all_chunks = []
    for topic in topics:
        base_metadata = {
            "url": topic.url or "",
            "group_ids": topic.group_ids,
            "group_titles": topic.group_titles,
            "linked_mentions": topic.linked_mentions,
        }
        for section in topic.sections:
            chunks = build_chunks_for_section(
                topic_title=topic.title,
                topic_id=topic.topic_id,
                section_name=section.name,
                section_text=section.text,
                base_metadata=base_metadata,
            )
            all_chunks.extend(chunks)

    # --- Summary stats ---
    chunk_lengths = [len(text) for text, _ in all_chunks]
    print(f"=== Summary ===")
    print(f"  Total chunks:   {len(all_chunks)}")
    print(f"  Avg chars:      {sum(chunk_lengths) // len(chunk_lengths)}")
    print(f"  Min chars:      {min(chunk_lengths)}")
    print(f"  Max chars:      {max(chunk_lengths)}")

    # Section tag distribution
    from collections import Counter
    tag_counts = Counter(meta["section_tag"] for _, meta in all_chunks)
    print(f"\n=== Section Tag Distribution ===")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")

    # --- Spot check on known topic ---
    print(f"\n=== Spot Check: '{SPOT_CHECK_TITLE}' ===")
    topic = next((t for t in topics if t.title.lower() == SPOT_CHECK_TITLE.lower()), None)
    if not topic:
        print(f"  WARNING: '{SPOT_CHECK_TITLE}' not found.")
        return

    base_metadata = {
        "url": topic.url or "",
        "group_ids": topic.group_ids,
        "group_titles": topic.group_titles,
        "linked_mentions": topic.linked_mentions,
    }

    for section in topic.sections:
        chunks = build_chunks_for_section(
            topic_title=topic.title,
            topic_id=topic.topic_id,
            section_name=section.name,
            section_text=section.text,
            base_metadata=base_metadata,
        )
        print(f"\n  Section '{section.name}' → {len(chunks)} chunk(s)")
        for text, meta in chunks:
            print(f"\n  --- Chunk {meta['chunk_index']} ({len(text)} chars) ---")
            print(f"  Metadata: section_tag={meta['section_tag']}, topic_id={meta['topic_id']}")
            print(f"  Text preview:\n  {text[:300]}...")


if __name__ == "__main__":
    main()