import re
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def extract_best_sentences(entity_text: str, passage: str, embed_model: SentenceTransformer, n_neighbors: int = 2) -> str:
    """Return best sentence + next n neighbors most relevant to the entity using query reformulation."""
    sentences = _split_sentences(passage)
    if len(sentences) <= 2:
        return passage

    query_vec = embed_model.encode(f"What is {entity_text}?")
    sentence_vecs = embed_model.encode(sentences)
    distances = [_cosine_distance(query_vec, vec) for vec in sentence_vecs]
    best_idx = int(np.argmin(distances))

    end = min(len(sentences), best_idx + n_neighbors + 1)
    return " ".join(sentences[best_idx:end]).strip()