import re
from typing import List
from dataclasses import dataclass

import nltk
from nltk.data import find
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from app.config import settings

try:
    find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    # Minimal fallback to keep extraction functional if corpus is unavailable.
    _STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with"}


@dataclass
class Entity:
    text: str
    label: str
    score: float
    start: int | None = None
    end: int | None = None


def load_ner_pipeline(model_name: str = settings.ner_model):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")


def _is_valid(text: str) -> bool:
    return not text.startswith("##")


def _normalize(text: str) -> str:
    """Strip leading stopwords — e.g. 'the allegra' → 'allegra'."""
    words = text.strip().split()
    while words and words[0].lower() in _STOPWORDS:
        words = words[1:]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def _is_meaningful(text: str) -> bool:
    return bool(text) and any(w.lower() not in _STOPWORDS for w in text.split())


def _deduplicate(entities: List[Entity]) -> List[Entity]:
    """Drop substring entities — keeps 'allergic rhinitis', drops 'allergic' and 'rhinitis'."""
    texts = [e.text.lower() for e in entities]
    return [
        e for i, e in enumerate(entities)
        if not any(texts[i] != texts[j] and texts[i] in texts[j] for j in range(len(texts)))
    ]


def normalize_entities(entities: List[Entity]) -> List[Entity]:
    seen = set()
    cleaned = []
    for e in entities:
        normalized = _normalize(e.text)
        if not _is_meaningful(normalized) or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        cleaned.append(Entity(
            text=normalized,
            label=e.label,
            score=e.score,
            start=e.start,
            end=e.end,
        ))
    return _deduplicate(cleaned)


def extract_entities(text: str, ner_pipeline, min_score: float = settings.ner_min_score) -> List[Entity]:
    results = ner_pipeline(text)
    entities = []
    for r in results:
        raw = r["word"].strip()
        if r["score"] < min_score or not _is_valid(raw):
            continue
        start = r.get("start")
        end = r.get("end")
        surface = raw
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            surface = text[start:end]
        entities.append(Entity(
            text=surface,
            label=r["entity_group"],
            score=round(r["score"], 3),
            start=start if isinstance(start, int) else None,
            end=end if isinstance(end, int) else None,
        ))
    return normalize_entities(entities)
