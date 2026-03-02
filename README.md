---
title: Medical Anchor
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Medical Anchor

A grounded medical information system for biomedical text analysis.

Given any text input — a clinical note, a social media post, a medical report — the system extracts biomedical entities and retrieves trusted, structured information about each one from official sources. Results are grounded strictly in retrieved evidence with full citations.

The system does not answer questions or generate text. It identifies what medical terms are present in your input and surfaces what MedlinePlus says about them.

---

## Data Source

**MedlinePlus Health Topics** — official XML corpus published by the U.S. National Library of Medicine. Covers thousands of health topics with curated summaries, categorized sections, and structured metadata.

---

## Architecture

**Pipeline 1 — Data (offline batch)**
```
MedlinePlus XML
→ Parsing + English filtering
→ HTML cleanup + metadata extraction (synonyms, see-references, MeSH, related topics)
→ Section-aware chunking (prefixed with topic title + section name for richer embeddings)
→ Embeddings + Chroma DB (small chunk embedded, full parent passage stored in metadata)
```

**Pipeline 2 — Online**
```
User input
→ Biomedical NER (samrawal/bert-base-uncased_clinical-ner)
→ Semantic search (top-5 candidates)
→ Validation gate: title / synonyms / see-references / MeSH matching
→ Best topic selection (field priority → distance)
→ Best chunk selection (closest embedding within winning topic)
→ Best sentence extraction (forward neighbor)
→ Grounded passage per entity + source URL
```

**Retrieval detail** — two-step approach to avoid pure semantic search limitations on short entity terms:
1. Validation gate fuzzy-matches entity against topic metadata. Priority: title > synonyms > see-references > MeSH. Falls back to distance threshold if no metadata match.
2. Once topic is selected, all its chunks are fetched and re-ranked by embedding distance to find the most relevant passage.

---

## Stack

- Python 3.11, Poetry
- sentence-transformers — BAAI/bge-small-en-v1.5
- transformers — samrawal/bert-base-uncased_clinical-ner
- Chroma, rapidfuzz, pydantic-settings
- FastAPI + uvicorn, Gradio + httpx
- Docker + supervisord (two-process single container)

## Models
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — MIT License
- [samrawal/bert-base-uncased_clinical-ner](https://huggingface.co/samrawal/bert-base-uncased_clinical-ner) — Apache 2.0

## API Endpoints

- `GET /health` — readiness check
- `POST /extract` — NER only
- `POST /retrieve` — full pipeline
- `POST /retrieve-from-entities` — retrieval only, skips NER

---

## Known Limitations

- **NER noise** — clinical NER occasionally extracts non-medical entities. Threshold tunable via `ner_min_score`.
- **Corpus gaps** — medications not covered by MedlinePlus topics. DailyMed extension planned.
- **Fuzzy matching** — `partial_ratio` can produce false positives on shared substrings. Switching to `token_sort_ratio` at threshold 75 noted as future improvement.
- **Cold start** — model loading ~60s on CPU. UI polls `/health` and disables Analyze button until ready.

---

## Optional Extensions

- DailyMed drug corpus (brand/generic medication coverage)
- MedlinePlus Connect API fallback
- Cross-encoder reranking
- Answer generation (LLM grounded in retrieved chunks)
- Evaluation pipeline
- Bilingual support (FR/EN)