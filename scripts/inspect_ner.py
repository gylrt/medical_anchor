from pathlib import Path
from app.ner import load_ner_pipeline, extract_entities

SAMPLES_DIR = Path("scripts/samples")

SAMPLE_TEXTS = {
    "allergic_rhinitis": (SAMPLES_DIR / "allergic_rhinitis.txt").read_text(encoding="utf-8"),
    "chest_pain": "The patient is a 58-year-old male with a history of type 2 diabetes, hypertension and chronic kidney disease presenting with chest pain and shortness of breath.",
    "oncology": "She was diagnosed with stage III breast cancer and is currently undergoing chemotherapy with doxorubicin and cyclophosphamide.",
}


def main():
    print("Loading NER pipeline...")
    ner = load_ner_pipeline()

    for name, text in SAMPLE_TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Sample: {name}")
        print(f"{'='*60}")

        entities = extract_entities(text, ner)

        if not entities:
            print("  No entities found.")
            continue

        by_label = {}
        for e in entities:
            by_label.setdefault(e.label, []).append(e)

        for label, ents in sorted(by_label.items()):
            print(f"\n  [{label}]")
            for e in ents:
                print(f"    {e.text} (score: {e.score})")


if __name__ == "__main__":
    main()