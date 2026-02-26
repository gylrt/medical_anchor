# Medical Anchor

A grounded medical insight system built on retrieval-augmented generation (RAG).

Given any text input — a question, a clinical note, or a social media post — the system extracts biomedical entities, retrieves trusted evidence from official sources, and returns concise, grounded answers strictly based on that evidence.

No hallucination. No unsourced claims. Answers are anchored to retrieved content only.

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
→ HTML cleanup + metadata extraction
→ Section-aware chunking
→ Embeddings + Chroma DB
```

**Pipeline 2 — Input processing (online)**
```
User input (question, report, article...)
→ Biomedical NER (d4data/biomedical-ner-all)
→ Extracted medical entities
```

**Combined — Grounded retrieval**
```
Entities + Chroma DB
→ Metadata-filtered vector retrieval
→ Relevant chunks with citations     ← standalone valuable output
→ Answer generation (optional, LLM grounded in retrieved chunks only)
```

Clean separation between:
- **Ingestion** — offline batch process
- **Retrieval API** — read-only, online service

---

## Stack

- Python 3.10
- Poetry (dependency management)
- sentence-transformers (embeddings)
- Chroma (vector store)
- transformers — d4data/biomedical-ner-all (NER)
- FastAPI (service layer)
- Docker (deployment)

---

## Project Status

### Pipeline 1 — Data (completed)
- [x] MedlinePlus XML download
- [x] Parsing + English filtering
- [x] HTML cleanup and text normalization
- [x] Synonyms and group metadata extraction
- [x] Linked mentions extraction
- [x] Section-aware chunking

### Pipeline 1 — Data (next)
- [ ] Chroma ingestion
- [ ] Inspection scripts

### Pipeline 2 — Input processing
- [ ] Biomedical NER (d4data/biomedical-ner-all)
- [ ] Entity extraction + normalization

### Combined — Grounded retrieval
- [ ] Entity-driven Chroma retrieval with metadata filtering
- [ ] Citation output (chunk + source URL)
- [ ] FastAPI service layer (`/health`, `/extract`, `/retrieve`)
- [ ] Docker deployment

### Optional extensions
- [ ] Answer generation (LLM grounded in retrieved chunks)
- [ ] Bilingual support (FR/EN)
- [ ] Topic graph expansion via linked mentions
- [ ] Reranker / cross-encoder
- [ ] Evaluation pipeline

---

## Project Structure

```
app/
  __init__.py
  download_medlineplus.py  # fetches MedlinePlus XML corpus
  parse_medlineplus.py     # XML parsing + metadata extraction
  chunking.py              # section-aware chunking
scripts/                   # inspection and verification scripts (coming)
data/                      # gitignored — XML corpus + vector DB
```

---

## Setup

```bash
# Create conda environment
conda create -n medical-rag python=3.10
conda activate medical-rag

# Install Poetry and dependencies
pip install poetry
poetry install

# Download the corpus
python app/download_medlineplus.py
```

---

## Design Philosophy

- **Grounding first** — answers reference retrieved evidence only
- **Entity-router pattern** — NER entities drive retrieval filtering, not pure semantic search
- **Metadata-aware** — section tags, topic groups, and linked mentions inform retrieval
- **Incremental engineering** — clean interfaces, testable steps, deployment-ready structure