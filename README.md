# Medical Anchor

A grounded medical information system for biomedical text analysis.

Given any text input — a clinical note, a social media post, a medical report — the system extracts biomedical entities and retrieves trusted, structured information about each one from official sources. Results are grounded strictly in retrieved evidence with full citations.

The system does not answer questions or generate text. It identifies what medical terms are present in your input and surfaces what MedlinePlus says about them.

---

## Data Source

**MedlinePlus Health Topics** — official XML corpus published by the U.S. National Library of Medicine.

Covers thousands of health topics with curated summaries, categorized sections, and structured metadata.

---

## Architecture

The system is built in two independent pipelines that combine at retrieval time.

**Pipeline 1 — Data (offline batch)**
```
MedlinePlus XML
→ Parsing + English filtering
→ HTML cleanup + metadata extraction (synonyms, see-references, MeSH, related topics)
→ Section-aware chunking
→ Embeddings + Chroma DB
```

**Pipeline 2 — Input processing (online)**
```
User input (clinical note, report, article...)
→ Biomedical NER (samrawal/bert-base-uncased_clinical-ner)
→ Entity extraction + stopword normalization + substring deduplication
```

**Combined — Grounded retrieval**
```
Extracted entities
→ Semantic search (top-5 candidates)
→ Validation gate: title / synonyms / see-references / MeSH matching
→ Best topic selection (field priority → distance)
→ Best chunk selection (closest embedding within winning topic)
→ Grounded passage per entity + source URL
```

Clean separation between:
- **Ingestion** — offline batch process
- **Retrieval API** — read-only, online service

---

## Retrieval Design

Retrieval uses a two-step approach to avoid pure semantic search limitations when matching short entity terms against long chunks:

1. **Topic selection** — semantic search returns top-5 candidates, then a validation gate checks if the entity fuzzy-matches the topic title, synonyms (`also-called`), see-references, or MeSH terms. Among valid candidates, priority order is: title > synonyms > see-references > MeSH. Distance breaks ties within the same priority level.

2. **Chunk selection** — once the best matching topic is identified, all chunks for that topic are fetched and the one closest to the entity embedding is returned.

Entities with no valid match above the distance threshold surface as "no relevant information found" rather than returning a wrong result.

## Stack

- Python 3.10
- Poetry (dependency management)
- sentence-transformers — BAAI/bge-small-en-v1.5 (embeddings)
- Chroma (vector store)
- transformers — samrawal/bert-base-uncased_clinical-ner (NER)
- rapidfuzz (fuzzy matching for validation gate)
- pydantic-settings (centralized configuration)
- FastAPI (service layer)
- Docker (deployment)

---

## Models
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — MIT License
- [samrawal/bert-base-uncased_clinical-ner](https://huggingface.co/samrawal/bert-base-uncased_clinical-ner) — Apache 2.0 — trained on i2b2/n2c2 2010 clinical NER dataset

---

## Project Status

### Pipeline 1 — Data (complete)
- [x] MedlinePlus XML download
- [x] Parsing + English filtering
- [x] HTML cleanup and text normalization
- [x] Synonyms, group metadata, MeSH headings, related topics extraction
- [x] Section-aware chunking
- [x] Chroma ingestion

### Pipeline 2 — Input processing (complete)
- [x] Biomedical NER (samrawal/bert-base-uncased_clinical-ner)
- [x] Entity extraction + stopword normalization + substring deduplication

### Combined — Grounded retrieval (complete)
- [x] Two-step retrieval: topic selection via validation gate + best chunk selection
- [x] Field priority matching: title > synonyms > see-references > MeSH
- [x] Grounded passage output per entity + source URL

### In progress
- [ ] FastAPI service layer (`/health`, `/extract`, `/retrieve`)
- [ ] Gradio UI (entity highlighting + grounded results display)
- [ ] Docker deployment
- [ ] Hugging Face Spaces deployment

### Optional extensions
- [ ] DailyMed drug corpus — add brand/generic drug information to cover medication entities currently returning no match. SPL XML format, deduplicate by active ingredient, ingest into same Chroma collection with `source: dailymed` metadata flag.
- [ ] MedlinePlus Connect API fallback — runtime fallback for entities not covered by local corpus
- [ ] Answer generation (LLM grounded in retrieved chunks)
- [ ] Bilingual support (FR/EN)
- [ ] Topic graph expansion via related topics + linked mentions
- [ ] Reranker / cross-encoder
- [ ] Evaluation pipeline

---

## Design Philosophy

- **Grounding first** — output references retrieved evidence only, no generation
- **Entity-router pattern** — NER entities drive retrieval filtering, not pure semantic search
- **Metadata-aware** — MeSH headings, section tags, topic groups, and linked mentions inform retrieval
- **Incremental engineering** — clean interfaces, testable steps, deployment-ready structure