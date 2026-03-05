import json
from pathlib import Path
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from app.ner import load_ner_pipeline, extract_entities
from app.retrieval import load_collections, retrieve_for_entities, _is_valid_match
from app.config import settings

SAMPLES_DIR = Path("scripts/samples")

SAMPLE_TEXTS = {
    "allergic_rhinitis": (SAMPLES_DIR / "allergic_rhinitis.txt").read_text(encoding="utf-8"),
    "chest_pain": "The patient is a 58-year-old male with a history of type 2 diabetes, hypertension and chronic kidney disease presenting with chest pain and shortness of breath.",
    "oncology": "She was diagnosed with stage III breast cancer and is currently undergoing chemotherapy with doxorubicin and cyclophosphamide.",
}


def _best_fuzzy_match(entity_text: str, topic_title: str, meta: dict) -> tuple:
    """Return (field, matched_term, score) for the best fuzzy match across all fields."""
    entity_lower = entity_text.lower()
    best = ("title", topic_title, fuzz.partial_ratio(entity_lower, topic_title.lower()))
    for field in ("mesh_terms", "synonyms", "see_references"):
        for term in json.loads(meta.get(field, "[]")):
            score = fuzz.partial_ratio(entity_lower, term.lower())
            if score > best[2]:
                best = (field, term, score)
    return best


def _target_collection_for_label(label: str, medline_collection, dailymed_collection):
    return dailymed_collection if (label or "").lower() == "treatment" else medline_collection


def show_candidates(entity_text: str, entity_label: str, medline_collection, dailymed_collection, embed_model):
    if (entity_label or "").lower() == "treatment":
        print("  candidates: lexical DailyMed match (no vector distance)")
        return

    collection = _target_collection_for_label(entity_label, medline_collection, dailymed_collection)
    query_vector = embed_model.encode(entity_text).tolist()
    hits = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "distances"],
    )
    print(f"  top 5 candidates:")
    for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
        valid = _is_valid_match(entity_text, meta)
        field, term, score = _best_fuzzy_match(entity_text, meta["topic_title"], meta)
        gate = f"{field} '{term}' {score}%" if valid else f"best fuzzy: {field} '{term}' {score}%"
        print(f"    {'✓' if valid else '✗'} dist={round(dist, 3)} — {meta['topic_title']} [{gate}]")


def main():
    print("Loading models...")
    ner = load_ner_pipeline()
    embed_model = SentenceTransformer(settings.embed_model)
    medline_collection, dailymed_collection = load_collections()

    for name, text in SAMPLE_TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Sample: {name}")
        print(f"{'='*60}")

        entities = extract_entities(text, ner)
        results = retrieve_for_entities(
            entities,
            medline_collection,
            dailymed_collection,
            embed_model,
        )

        for r in results:
            print(f"\n  [{r.entity.label}] {r.entity.text}")
            show_candidates(
                r.entity.text,
                r.entity.label,
                medline_collection,
                dailymed_collection,
                embed_model,
            )

            if not r.matched:
                print(f"  → no relevant information found")
                continue

            meta = {
                "mesh_terms": r.mesh_terms or "[]",
                "synonyms": r.synonyms or "[]",
                "see_references": r.see_references or "[]",
            }
            field, term, score = _best_fuzzy_match(r.entity.text, r.topic_title, meta)
            print(f"  → {r.topic_title} (dist={r.distance}, {field} '{term}' {score}%)")
            if r.generic_name:
                print(f"  → generic: {r.generic_name}")
            print(f"  → {r.source_url}")
            print(f"  → {r.passage[:200]}...")


if __name__ == "__main__":
    main()
