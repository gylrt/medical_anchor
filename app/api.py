import re
import html
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ner import load_ner_pipeline, extract_entities, Entity
from app.retrieval import load_collections, retrieve_for_entities
from app.utils import extract_best_sentences


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ner_pipeline = load_ner_pipeline()
    app.state.embed_model = SentenceTransformer(settings.embed_model)
    app.state.medline_collection, app.state.dailymed_collection = load_collections()
    yield


app = FastAPI(
    title="Medical Anchor",
    description="Grounded medical entity extraction and retrieval from MedlinePlus.",
    version="0.1.0",
    lifespan=lifespan,
)


class TextInput(BaseModel):
    text: str


class EntityResponse(BaseModel):
    text: str
    label: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None


class EntitiesInput(BaseModel):
    entities: List[EntityResponse]


class RetrievalItem(BaseModel):
    entity: EntityResponse
    matched: bool
    topic_title: Optional[str] = None
    source_url: Optional[str] = None
    generic_name: Optional[str] = None
    passage: Optional[str] = None
    best_sentences: Optional[str] = None
    highlighted_passage: Optional[str] = None
    highlighted_best_sentences: Optional[str] = None
    distance: Optional[float] = None


class ExtractResponse(BaseModel):
    entities: List[EntityResponse]


class RetrieveResponse(BaseModel):
    results: List[RetrievalItem]


def _validate_text(text: str):
    if not text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")
    if len(text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail=f"Input text too long. Maximum {settings.max_text_length} characters.")


def _highlight(entity_text: str, text: str) -> str:
    safe_text = html.escape(text or "")
    safe_entity = html.escape(entity_text or "")
    if not safe_text or not safe_entity:
        return safe_text
    pattern = re.compile(r"\b" + re.escape(safe_entity) + r"\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"<strong>{m.group()}</strong>", safe_text)


def _build_retrieval_response(entities: List[Entity], app_state) -> RetrieveResponse:
    if not entities:
        return RetrieveResponse(results=[])
    results = retrieve_for_entities(
        entities,
        app_state.medline_collection,
        app_state.dailymed_collection,
        app_state.embed_model,
    )
    items = []
    for r in results:
        best_sentences, highlighted_passage, highlighted_best = None, None, None
        if r.matched and r.passage:
            best_sentences = extract_best_sentences(r.entity.text, r.passage, app_state.embed_model)
            highlighted_passage = _highlight(r.entity.text, r.passage)
            highlighted_best = _highlight(r.entity.text, best_sentences)
        items.append(RetrievalItem(
            entity=EntityResponse(
                text=r.entity.text,
                label=r.entity.label,
                score=r.entity.score,
                start=r.entity.start,
                end=r.entity.end,
            ),
            matched=r.matched,
            topic_title=r.topic_title,
            source_url=r.source_url,
            generic_name=r.generic_name,
            passage=r.passage,
            best_sentences=best_sentences,
            highlighted_passage=highlighted_passage,
            highlighted_best_sentences=highlighted_best,
            distance=r.distance,
        ))
    return RetrieveResponse(results=items)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "medlineplus_count": app.state.medline_collection.count(),
        "dailymed_count": app.state.dailymed_collection.count(),
    }


@app.post("/extract", response_model=ExtractResponse)
def extract(body: TextInput):
    _validate_text(body.text)
    entities = extract_entities(body.text, app.state.ner_pipeline)
    return ExtractResponse(
        entities=[
            EntityResponse(
                text=e.text,
                label=e.label,
                score=e.score,
                start=e.start,
                end=e.end,
            )
            for e in entities
        ]
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(body: TextInput):
    """Full pipeline — extracts entities then retrieves. Useful for direct API calls."""
    _validate_text(body.text)
    entities = extract_entities(body.text, app.state.ner_pipeline)
    return _build_retrieval_response(entities, app.state)


@app.post("/retrieve-from-entities", response_model=RetrieveResponse)
def retrieve_from_entities(body: EntitiesInput):
    """Retrieval only — skips NER, uses pre-extracted entities from /extract."""
    if not body.entities:
        return RetrieveResponse(results=[])
    entities = [
        Entity(
            text=e.text,
            label=e.label,
            score=e.score,
            start=e.start,
            end=e.end,
        )
        for e in body.entities
    ]
    return _build_retrieval_response(entities, app.state)
