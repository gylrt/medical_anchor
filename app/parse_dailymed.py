from dataclasses import dataclass
from typing import List
import html
import re
from io import BytesIO

from lxml import etree


HL7_NS = {"hl7": "urn:hl7-org:v3"}

# Top-level DailyMed section codes to extract.
TARGET_SECTION_CODES = {
    "34066-1": "boxed_warning",
    "34067-9": "indications_and_usage",
}

# Section behavior by top-level code.
EXPAND_CHILD_SECTIONS_CODES = {
    "34067-9",  # indications and usage
}
EXCERPT_FIRST_CODES = {
    "34066-1",  # boxed warning
}
CAPPED_SECTION_CODES = {
    "34066-1",  # boxed warning
}


@dataclass
class DrugNameCode:
    name: str
    code: str
    name_type: str  # brand | generic | active_moiety
    normalized_name: str


@dataclass
class DailyMedSection:
    section_key: str
    code: str
    title: str
    text: str


@dataclass
class DailyMedLabel:
    document_id: str
    set_id: str
    version: str
    effective_time: str
    source_url: str
    drug_name_codes: List[DrugNameCode]
    synonyms: List[str]
    sections: List[DailyMedSection]


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_name(s: str) -> str:
    s = _clean_text(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _get_text(elem) -> str:
    if elem is None:
        return ""
    return _clean_text(" ".join(elem.itertext()))


def _attr(elem, name: str) -> str:
    if elem is None:
        return ""
    return _clean_text(elem.get(name) or "")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _dedupe_name_codes(items: List[DrugNameCode]) -> List[DrugNameCode]:
    seen = set()
    out: List[DrugNameCode] = []
    for item in items:
        key = (item.normalized_name, item.code, item.name_type)
        if item.normalized_name and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _extract_section_text(section_elem, prefer_excerpt: bool = True) -> str:
    excerpt_chunks: List[str] = []
    for excerpt_text in section_elem.findall("./hl7:excerpt//hl7:text", HL7_NS):
        txt = _get_text(excerpt_text)
        if txt:
            excerpt_chunks.append(txt)

    chunks: List[str] = []
    main_text = section_elem.find("./hl7:text", HL7_NS)
    if main_text is not None:
        txt = _get_text(main_text)
        if txt:
            chunks.append(txt)

    if prefer_excerpt:
        if excerpt_chunks:
            return " ".join(_dedupe_keep_order(excerpt_chunks))
        if chunks:
            return " ".join(_dedupe_keep_order(chunks))
    else:
        if chunks:
            return " ".join(_dedupe_keep_order(chunks))
        if excerpt_chunks:
            return " ".join(_dedupe_keep_order(excerpt_chunks))

    return ""


def _iter_section_tree(section_elem):
    yield section_elem
    for child in section_elem.findall("./hl7:component/hl7:section", HL7_NS):
        yield from _iter_section_tree(child)


def parse_dailymed_labels(
    xml_path: str,
    max_boxed_warning_chars: int = 2500,
) -> List[DailyMedLabel]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(xml_path, parser)
    return _parse_labels_from_root(tree.getroot(), max_boxed_warning_chars=max_boxed_warning_chars)


def parse_dailymed_labels_from_xml_bytes(
    xml_bytes: bytes,
    max_boxed_warning_chars: int = 2500,
) -> List[DailyMedLabel]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(BytesIO(xml_bytes), parser)
    return _parse_labels_from_root(tree.getroot(), max_boxed_warning_chars=max_boxed_warning_chars)


def _parse_labels_from_root(
    root,
    max_boxed_warning_chars: int = 2500,
) -> List[DailyMedLabel]:
    document_id = _attr(root.find("./hl7:id", HL7_NS), "root")
    set_id = _attr(root.find("./hl7:setId", HL7_NS), "root")
    version = _attr(root.find("./hl7:versionNumber", HL7_NS), "value")
    effective_time = _attr(root.find("./hl7:effectiveTime", HL7_NS), "value")
    source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else ""

    product_sections = root.xpath(
        ".//hl7:section[hl7:code[@code='48780-1']]",
        namespaces=HL7_NS,
    )
    product_section = product_sections[0] if product_sections else None

    name_codes: List[DrugNameCode] = []
    synonyms: List[str] = []

    if product_section is not None:
        manufactured_products = product_section.findall(
            ".//hl7:subject/hl7:manufacturedProduct/hl7:manufacturedProduct",
            HL7_NS,
        )
        for mp in manufactured_products:
            product_code = _attr(mp.find("./hl7:code", HL7_NS), "code")

            brand_name = _get_text(mp.find("./hl7:name", HL7_NS))
            if brand_name:
                synonyms.append(brand_name)
                name_codes.append(
                    DrugNameCode(
                        name=brand_name,
                        code=product_code,
                        name_type="brand",
                        normalized_name=_normalize_name(brand_name),
                    )
                )

            for g in mp.findall(".//hl7:asEntityWithGeneric/hl7:genericMedicine/hl7:name", HL7_NS):
                generic_name = _get_text(g)
                if generic_name:
                    synonyms.append(generic_name)
                    name_codes.append(
                        DrugNameCode(
                            name=generic_name,
                            code=product_code,
                            name_type="generic",
                            normalized_name=_normalize_name(generic_name),
                        )
                    )

            for moiety in mp.findall(".//hl7:ingredient[@classCode='ACTIB']//hl7:activeMoiety//hl7:name", HL7_NS):
                moiety_name = _get_text(moiety)
                if moiety_name:
                    synonyms.append(moiety_name)
                    name_codes.append(
                        DrugNameCode(
                            name=moiety_name,
                            code=product_code,
                            name_type="active_moiety",
                            normalized_name=_normalize_name(moiety_name),
                        )
                    )

    top_sections = root.findall("./hl7:component/hl7:structuredBody/hl7:component/hl7:section", HL7_NS)
    sections: List[DailyMedSection] = []
    for top in top_sections:
        top_code = _attr(top.find("./hl7:code", HL7_NS), "code")
        if top_code not in TARGET_SECTION_CODES:
            continue

        # Section behavior is configured by code.
        nodes = list(_iter_section_tree(top)) if top_code in EXPAND_CHILD_SECTIONS_CODES else [top]
        for sec in nodes:
            sec_code = _attr(sec.find("./hl7:code", HL7_NS), "code")
            prefer_excerpt = top_code in EXCERPT_FIRST_CODES
            text = _extract_section_text(sec, prefer_excerpt=prefer_excerpt)
            if not text:
                continue

            if top_code in CAPPED_SECTION_CODES and max_boxed_warning_chars > 0 and len(text) > max_boxed_warning_chars:
                text = text[:max_boxed_warning_chars].rstrip()

            sections.append(
                DailyMedSection(
                    section_key=TARGET_SECTION_CODES[top_code],
                    code=sec_code,
                    title=_get_text(sec.find("./hl7:title", HL7_NS)),
                    text=text,
                )
            )

    label = DailyMedLabel(
        document_id=document_id,
        set_id=set_id,
        version=version,
        effective_time=effective_time,
        source_url=source_url,
        drug_name_codes=_dedupe_name_codes(name_codes),
        synonyms=_dedupe_keep_order(synonyms),
        sections=sections,
    )
    return [label]
