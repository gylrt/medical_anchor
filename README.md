# Medical Anchor

A grounded medical information system for biomedical text analysis.

### **[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/gylrt/medical-anchor)**

Given any text input — a clinical note, a social media post, a medical report — the system extracts biomedical entities and retrieves trusted, structured information about each one from official sources. Results are grounded strictly in retrieved evidence with full citations.

The system does not answer questions or generate text. It identifies what medical terms are present in your input and surfaces grounded evidence from MedlinePlus and DailyMed.

---

## Data Sources

- **MedlinePlus Health Topics** - official XML corpus published by the U.S. National Library of Medicine. Covers thousands of health topics with curated summaries, categorized sections, and structured metadata.
- **DailyMed SPL** - official Structured Product Label corpus from the U.S. National Library of Medicine, used for medication/treatment label data (brand/generic aliases, indications text, source URLs).


---

## Architecture

**Pipeline 1 — Data (offline batch)**
```
MedlinePlus XML / DailyMed SPL ZIPs
→ Source-specific parsing + normalization
→ Metadata extraction (titles, synonyms/aliases, references, source URL)
→ Section-aware chunking (prefixed with topic title + section name for richer embeddings)
→ Embeddings + Chroma ingestion (per-source collections)
→ DailyMed name index generation (for fast lexical treatment matching)
```

**Pipeline 2 — Online**
```
User input
→ Biomedical NER (samrawal/bert-base-uncased_clinical-ner)
→ Entity routing by label:
   - problem/test -> MedlinePlus retrieval
   - treatment -> DailyMed retrieval
→ Source-specific ranking:
   - MedlinePlus: semantic search, top topic chunk selection
   - DailyMed: lexical name-index match + topic chunk selection
→ Grounded passage per entity + source URL
```

**MedlinePlus Retrieval detail** — two-step approach to avoid pure semantic search limitations on short entity terms:
1. Validation gate fuzzy-matches entity against topic metadata. Priority: title > synonyms > see-references > MeSH. Falls back to distance threshold if no metadata match.
2. Once topic is selected, all its chunks are fetched and re-ranked by embedding distance to find the most relevant passage.

**Dual-source routing (current)**:
1. `problem` and `test` entities route to **MedlinePlus** (vector retrieval + metadata gate).
2. `treatment` entities route to **DailyMed** (lexical name-index match: normalized names > title > synonyms, then best chunk by topic).

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
- **Fuzzy matching** — `partial_ratio` can produce false positives on shared substrings. Switching to `token_sort_ratio` at threshold 75 noted as future improvement.
- **Cold start** — model loading ~60s on CPU. UI polls `/health` and disables Analyze button until ready.

---

## Optional Extensions

- Answer generation (LLM grounded in retrieved chunks)
- MedlinePlus Connect API fallback
- Cross-encoder reranking
- Evaluation pipeline
- Bilingual support (FR/EN)