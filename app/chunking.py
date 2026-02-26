from typing import List, Dict, Tuple
import re


def normalize_section_tag(section_name: str) -> str:
    """Map section name to a canonical tag for retrieval filtering."""
    s = section_name.strip().lower()
    if "summary" in s or "overview" in s:
        return "overview"
    return "other"


def split_into_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries. May break on abbreviations like Dr. or e.g."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_blocks(
    text: str,
    min_chars: int = 80,
    max_chars: int = 500,
) -> List[Tuple[str, str]]:
    """
    Split text into (block, parent_passage) tuples.

    Keeping blocks small (max 500 chars) improves retrieval precision —
    each block covers one focused idea. parent_passage includes neighboring
    blocks so the retrieval layer can return readable context without an LLM.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    raw_blocks: List[str] = []
    buf_sentences: List[str] = []
    buf_len = 0

    for sent in sentences:
        if buf_len + len(sent) <= max_chars:
            buf_sentences.append(sent)
            buf_len += len(sent)
        else:
            if buf_sentences:
                raw_block = " ".join(buf_sentences)
                if len(raw_block) >= min_chars:
                    raw_blocks.append(raw_block)
            buf_sentences = [sent]
            buf_len = len(sent)

    if buf_sentences:
        raw_block = " ".join(buf_sentences)
        if len(raw_block) >= min_chars:
            raw_blocks.append(raw_block)

    # Each block gets its immediate neighbors as context window
    result: List[Tuple[str, str]] = []
    for i, raw_block in enumerate(raw_blocks):
        prev = raw_blocks[i - 1] if i > 0 else ""
        nxt = raw_blocks[i + 1] if i < len(raw_blocks) - 1 else ""
        parent = " ".join(filter(None, [prev, raw_block, nxt]))
        result.append((raw_block, parent))

    return result


def build_chunks_for_section(
    topic_title: str,
    topic_id: str,
    section_name: str,
    section_text: str,
    base_metadata: Dict,
) -> List[Tuple[str, Dict]]:
    """
    Returns (chunk_text, metadata) tuples for one topic section.

    Title/Section prefix is prepended to the chunk text to anchor the
    embedding — improves retrieval quality for short or generic passages.
    parent_passage is stored in metadata for display at retrieval time.
    """
    section_tag = normalize_section_tag(section_name)
    blocks = split_into_blocks(section_text)

    out = []
    for i, (raw_block, parent_passage) in enumerate(blocks):
        chunk_text = f"Title: {topic_title}\nSection: {section_name}\n\n{raw_block}"
        meta = {
            **base_metadata,
            "topic_title": topic_title,
            "topic_id": topic_id,
            "section_name": section_name,
            "section_tag": section_tag,
            "chunk_index": i,
            "parent_passage": parent_passage,
        }
        out.append((chunk_text, meta))
    return out