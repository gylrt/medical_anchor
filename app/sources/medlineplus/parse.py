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
class MeshHeading:
    mesh_id: str
    term: str


@dataclass
class RelatedTopic:
    topic_id: str
    title: str
    url: str


@dataclass
class Topic:
    topic_id: str
    title: str
    url: Optional[str]
    synonyms: List[str]
    see_references: List[str]
    sections: List[TopicSection]
    language: str
    group_ids: List[str]
    group_titles: List[str]
    linked_mentions: List[str]      # anchor texts from summary <a> links
    mesh_headings: List[MeshHeading]  # MeSH terms for entity-driven filtering
    related_topics: List[RelatedTopic]  # topic graph for future retrieval expansion


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_text(elem) -> str:
    if elem is None:
        return ""
    raw = " ".join(elem.itertext())
    return _clean_text(raw)


def _attr(elem, name: str) -> str:
    if elem is None:
        return ""
    return _clean_text(elem.get(name) or "")


def _clean_html_and_extract_anchor_texts(html_str: str) -> Tuple[str, List[str]]:
    """
    Convert HTML summary to plain text, extracting anchor texts as linked mentions.
    Linked mentions are a lightweight graph signal for future retrieval expansion.
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
        a.replace_with(txt)

    plain = soup.get_text(" ", strip=True)
    plain = re.sub(r"\s+", " ", plain).strip()

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

    Extracts:
      - Summary text (plain, embedding-ready)
      - Group metadata, synonyms, linked mentions
      - See-references
      - MeSH headings — controlled vocabulary for entity-driven filtering
      - Related topics — topic graph for future retrieval expansion
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    topics: List[Topic] = []

    for ht in root.findall(".//health-topic"):
        lang = _attr(ht, "language").lower()
        url_attr = _attr(ht, "url")

        if english_only:
            if lang and lang != "english":
                continue
            if url_attr and "/spanish/" in url_attr.lower():
                continue

        topic_id = _attr(ht, "id") or _get_text(ht.find("./id"))
        title = _attr(ht, "title") or _get_text(ht.find("./title"))
        url = url_attr or _get_text(ht.find("./url")) or None

        if not title:
            continue

        # ---- synonyms (also-called field)----
        synonyms: List[str] = []
        for ac in ht.findall(".//also-called"):
            txt = _get_text(ac)
            if txt:
                synonyms.append(txt)
        synonyms = list(dict.fromkeys(synonyms))

        # ---- see-references (redirect terms) ----
        see_references: List[str] = []
        for sr in ht.findall(".//see-reference"):
            txt = _get_text(sr)
            if txt:
                see_references.append(txt)
        see_references = list(dict.fromkeys(see_references))

        # ---- groups ----
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

        # ---- MeSH headings ----
        mesh_headings: List[MeshHeading] = []
        for mh in ht.findall("./mesh-heading"):
            descriptor = mh.find("./descriptor")
            if descriptor is not None:
                mesh_id = _attr(descriptor, "id")
                term = _get_text(descriptor)
                if term:
                    mesh_headings.append(MeshHeading(mesh_id=mesh_id, term=term))

        # ---- related topics ----
        related_topics: List[RelatedTopic] = []
        for rt in ht.findall("./related-topic"):
            rt_id = _attr(rt, "id")
            rt_url = _attr(rt, "url")
            rt_title = _get_text(rt)
            if rt_title:
                related_topics.append(RelatedTopic(
                    topic_id=rt_id,
                    title=rt_title,
                    url=rt_url,
                ))

        # ---- summary ----
        sections: List[TopicSection] = []
        linked_mentions: List[str] = []
        full_summary_raw = _get_text(ht.find("./full-summary"))
        short_summary_raw = _get_text(ht.find("./summary"))
        summary_raw = full_summary_raw or short_summary_raw

        if summary_raw:
            summary_text, linked_mentions = _clean_html_and_extract_anchor_texts(summary_raw)
            if summary_text:
                sections.append(TopicSection(name="Summary", text=summary_text))

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
                see_references=see_references,
                sections=sections,
                language=(lang or "english") if english_only else (lang or "unknown"),
                group_ids=group_ids,
                group_titles=group_titles,
                linked_mentions=linked_mentions,
                mesh_headings=mesh_headings,
                related_topics=related_topics,
            )
        )

    return topics
