from dataclasses import dataclass
from typing import List, Optional, Tuple
from lxml import etree
import html
import re

from bs4 import BeautifulSoup


@dataclass
class TopicSection:
    name: str
    text: str


@dataclass
class Topic:
    topic_id: str
    title: str
    url: Optional[str]
    synonyms: List[str]
    sections: List[TopicSection]
    language: str  # e.g., "english"
    group_ids: List[str]
    group_titles: List[str]
    linked_mentions: List[str]  # anchor texts from <a> links in summaries (lowercased)


def _clean_text(s: str) -> str:
    """
    Unescape HTML entities and normalize whitespace.
    Example: '&lt;p&gt;Hello&lt;/p&gt;' -> '<p>Hello</p>'
    """
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_text(elem) -> str:
    """
    Collect all text from an XML element (including nested children),
    unescape HTML entities, and normalize whitespace.
    """
    if elem is None:
        return ""
    raw = " ".join(elem.itertext())
    return _clean_text(raw)


def _attr(elem, name: str) -> str:
    """
    Safe attribute getter with cleanup.
    """
    if elem is None:
        return ""
    return _clean_text(elem.get(name) or "")


def _clean_html_and_extract_anchor_texts(html_str: str) -> Tuple[str, List[str]]:
    """
    Convert HTML content to clean plain text for embeddings,
    while extracting anchor texts (linked mentions) as a lightweight graph signal.

    - Replaces <a>...</a> with visible anchor text (drops URL)
    - Returns (plain_text, linked_mentions)
    """
    if not html_str:
        return "", []

    soup = BeautifulSoup(html_str, "html.parser")

    anchor_texts: List[str] = []
    for a in soup.find_all("a"):
        txt = a.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            anchor_texts.append(txt.lower())
        # Replace the link tag with just its visible text
        a.replace_with(txt)

    plain = soup.get_text(" ", strip=True)
    plain = re.sub(r"\s+", " ", plain).strip()

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for t in anchor_texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return plain, deduped


def parse_medlineplus_topics(xml_path: str, english_only: bool = True) -> List[Topic]:
    """
    Parse MedlinePlus Health Topics XML into normalized Topic objects.

    Key behaviors:
      - Filters to English topics
      - Extracts multiple <group> entries (ids + titles) as metadata
      - Converts summary HTML to plain text for embeddings
      - Extracts anchor texts from summary links into linked_mentions
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    topics: List[Topic] = []

    for ht in root.findall(".//health-topic"):
        # ---- language filter ----
        lang = _attr(ht, "language").lower()  # e.g., "english", "spanish"
        url_attr = _attr(ht, "url")

        if english_only:
            if lang and lang != "english":
                continue
            if url_attr and "/spanish/" in url_attr.lower():
                continue

        # ---- core fields (attributes-first, fallback to child tags) ----
        topic_id = _attr(ht, "id") or _get_text(ht.find("./id"))
        title = _attr(ht, "title") or _get_text(ht.find("./title"))
        url = url_attr or _get_text(ht.find("./url")) or None

        if not title:
            continue

        # ---- synonyms / also called ----
        synonyms: List[str] = []
        for ac in ht.findall(".//also-called"):
            txt = _get_text(ac)
            if txt:
                synonyms.append(txt)
        synonyms = list(dict.fromkeys(synonyms))  # dedupe preserve order

        # ---- group metadata (multiple <group> entries) ----
        group_ids: List[str] = []
        group_titles: List[str] = []
        for g in ht.findall("./group"):
            gid = _attr(g, "id")
            gtitle = _get_text(g)
            if gid:
                group_ids.append(gid)
            if gtitle:
                group_titles.append(gtitle)
        group_ids = list(dict.fromkeys(group_ids))
        group_titles = list(dict.fromkeys(group_titles))

        # ---- sections (summary only) ----
        sections: List[TopicSection] = []
        linked_mentions: List[str] = []
        full_summary_raw = _get_text(ht.find("./full-summary"))
        short_summary_raw = _get_text(ht.find("./summary"))
        summary_raw = full_summary_raw or short_summary_raw

        if summary_raw:
            summary_text, linked_mentions = _clean_html_and_extract_anchor_texts(summary_raw)
            if summary_text:
                sections.append(TopicSection(name="Summary", text=summary_text))

        # Fallback: if no summary found, store a "Body" section
        if not sections:
            body = _get_text(ht)
            if body and len(body) > 80:
                sections.append(TopicSection(name="Body", text=body))

        topics.append(
            Topic(
                topic_id=topic_id or title,
                title=title,
                url=url,
                synonyms=synonyms,
                sections=sections,
                language=(lang or "english") if english_only else (lang or "unknown"),
                group_ids=group_ids,
                group_titles=group_titles,
                linked_mentions=linked_mentions,
            )
        )

    return topics