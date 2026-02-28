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
Extracted entities + Chroma DB
→ Metadata-filtered vector retrieval
→ Grounded passages per entity + source URL
```

Clean separation between:
- **Ingestion** — offline batch process
- **Retrieval API** — read-only, online service

---

## Stack

- Python 3.10
- Poetry (dependency management)
- sentence-transformers — BAAI/bge-small-en-v1.5 (embeddings)
- Chroma (vector store)
- transformers — d4data/biomedical-ner-all (NER)
- FastAPI (service layer)
- Docker (deployment)

---

## Models
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — MIT License
- [d4data/biomedical-ner-all](https://huggingface.co/d4data/biomedical-ner-all) — MIT License

---

## Project Status

### Pipeline 1 — Data (complete)
- [x] MedlinePlus XML download
- [x] Parsing + English filtering
- [x] HTML cleanup and text normalization
- [x] Synonyms, group metadata, MeSH headings, related topics extraction
- [x] Section-aware chunking
- [x] Chroma ingestion

### Pipeline 2 — Input processing
- [ ] Biomedical NER (d4data/biomedical-ner-all)
- [ ] Entity extraction + normalization

### Combined — Grounded retrieval
- [ ] Entity-driven Chroma retrieval with metadata filtering
- [ ] Grounded passage output per entity + source URL
- [ ] FastAPI service layer (`/health`, `/extract`, `/retrieve`)
- [ ] Gradio UI (entity highlighting + grounded results display)
- [ ] Docker deployment (Chroma DB baked into image)
- [ ] Hugging Face Spaces deployment

### Optional extensions
- [ ] Answer generation (LLM grounded in retrieved chunks)
- [ ] Bilingual support (FR/EN)
- [ ] Topic graph expansion via related topics + linked mentions
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
  ingest.py                # embedding + Chroma ingestion
scripts/
  inspect_download.py      # verify downloaded XML and manifest
  inspect_parser.py        # verify parser output and field completeness
  inspect_chunking.py      # verify chunk sizes and metadata
  inspect_ingest.py        # verify Chroma collection state
data/                      # gitignored — XML corpus + Chroma DB
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

# Run ingestion
python app/ingest.py
```

---

## Verification

Run inspect scripts in order to verify each step:

```bash
python scripts/inspect_download.py
python scripts/inspect_parser.py
python scripts/inspect_chunking.py
python scripts/inspect_ingest.py
```

---

## Design Philosophy

- **Grounding first** — output references retrieved evidence only, no generation
- **Entity-router pattern** — NER entities drive retrieval filtering, not pure semantic search
- **Metadata-aware** — MeSH headings, section tags, topic groups, and linked mentions inform retrieval
- **Incremental engineering** — clean interfaces, testable steps, deployment-ready structure