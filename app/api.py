import re
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ner import load_ner_pipeline, extract_entities, Entity
from app.retrieval import load_collection, retrieve_for_entities


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ner_pipeline = load_ner_pipeline()
    app.state.embed_model = SentenceTransformer(settings.embed_model)
    app.state.collection = load_collection()
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


class RetrievalItem(BaseModel):
    entity: EntityResponse
    matched: bool
    topic_title: Optional[str] = None
    source_url: Optional[str] = None
    passage: Optional[str] = None
    highlighted_passage: Optional[str] = None
    distance: Optional[float] = None


class ExtractResponse(BaseModel):
    entities: List[EntityResponse]


class RetrieveResponse(BaseModel):
    results: List[RetrievalItem]


def _highlight(entity_text: str, passage: str) -> str:
    pattern = re.compile(r"\b" + re.escape(entity_text) + r"\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"<strong>{m.group()}</strong>", passage)


@app.get("/health")
def health():
    return {"status": "ok", "collection_count": app.state.collection.count()}


@app.post("/extract", response_model=ExtractResponse)
def extract(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")
    if len(body.text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail=f"Input text too long. Maximum {settings.max_text_length} characters.")
    entities: List[Entity] = extract_entities(body.text, app.state.ner_pipeline)
    return ExtractResponse(
        entities=[EntityResponse(text=e.text, label=e.label, score=e.score) for e in entities]
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")
    if len(body.text) > settings.max_text_length:
        raise HTTPException(status_code=422, detail=f"Input text too long. Maximum {settings.max_text_length} characters.")
    entities = extract_entities(body.text, app.state.ner_pipeline)
    if not entities:
        return RetrieveResponse(results=[])
    results = retrieve_for_entities(entities, app.state.collection, app.state.embed_model)
    return RetrieveResponse(
        results=[
            RetrievalItem(
                entity=EntityResponse(text=r.entity.text, label=r.entity.label, score=r.entity.score),
                matched=r.matched,
                topic_title=r.topic_title,
                source_url=r.source_url,
                passage=r.passage,
                highlighted_passage=_highlight(r.entity.text, r.passage) if r.matched and r.passage else None,
                distance=r.distance,
            )
            for r in results
        ]
    )