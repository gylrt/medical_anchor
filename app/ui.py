import re
import httpx
import gradio as gr

API_BASE = "http://localhost:8000"

CSS = """
.annotated { font-size: 1rem; line-height: 2.4; color: var(--body-text-color); }
.entity {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 4px; padding: 1px 7px; margin: 0 2px;
    font-weight: 500; color: #111; cursor: pointer; text-decoration: none;
}
.entity .label {
    font-size: 0.72em; font-weight: 400; color: #555;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.matched   { background: #d1fae5; }
.unmatched { background: #ffedd5; }
.legend { display: flex; gap: 16px; margin-bottom: 14px; font-size: 0.85em; color: var(--body-text-color); }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 12px; height: 12px; border-radius: 2px; }
.card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px; padding: 20px; margin-bottom: 12px;
    color: var(--body-text-color);
}
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.card-entity { font-weight: 700; font-size: 1.3rem; color: var(--body-text-color); }
.card-label {
    font-size: 0.75em; text-transform: uppercase; color: var(--body-text-color);
    border: 1px solid var(--border-color-primary); border-radius: 3px; padding: 2px 6px;
    opacity: 0.6;
}
.card-topic { font-size: 0.95em; color: var(--body-text-color); margin-bottom: 12px; }
.card-topic span { font-weight: 700; font-size: 1.05em; }
.card-passage { font-size: 0.95em; color: var(--body-text-color); line-height: 1.7; }
.card-passage strong { font-weight: 700; color: var(--body-text-color); }
details { margin-top: 10px; }
summary {
    font-size: 0.9em; color: var(--link-text-color, #2563eb); cursor: pointer;
    user-select: none; width: fit-content; list-style: none; display: inline-block;
}
.card details summary::before { content: "↓ Show full passage"; }
.card details[open] summary::before { content: "↑ Hide full passage"; }
.full-passage {
    font-size: 0.9em; color: var(--body-text-color); line-height: 1.7;
    margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-color-primary);
    opacity: 0.75;
}
.full-passage strong { font-weight: 700; opacity: 1; }
.card-url { margin-top: 12px; }
.card-url a { font-size: 0.85em; color: var(--link-text-color, #2563eb); text-decoration: none; }
.card-url a:hover { text-decoration: underline; }
.no-match { color: var(--body-text-color); font-size: 0.9em; font-style: italic; opacity: 0.5; }
.unmatched-toggle {
    font-size: 0.9em; color: var(--body-text-color); cursor: pointer;
    list-style: none; opacity: 0.5; margin-top: 8px; display: inline-block;
}
.unmatched-toggle:hover { opacity: 0.8; }
"""


def _card_id(entity_text: str) -> str:
    return "card-" + re.sub(r'\W+', '-', entity_text.lower()).strip('-')


def _build_annotation_html(input_text: str, results: list) -> str:
    matched_map = {r["entity"]["text"]: r["matched"] for r in results}
    label_map = {r["entity"]["text"]: r["entity"]["label"] for r in results}
    entities_sorted = sorted(matched_map.keys(), key=len, reverse=True)

    annotated = input_text
    placeholders = {}
    for i, entity_text in enumerate(entities_sorted):
        status = "matched" if matched_map[entity_text] else "unmatched"
        label = label_map.get(entity_text, "")
        if matched_map[entity_text]:
            chip = (
                f'<a class="entity {status}" href="#{_card_id(entity_text)}">'
                f'{entity_text} <span class="label">{label}</span>'
                f'</a>'
            )
        else:
            chip = (
                f'<span class="entity {status}">'
                f'{entity_text} <span class="label">{label}</span>'
                f'</span>'
            )
        placeholder = f"__ENTITY_{i}__"
        placeholders[placeholder] = chip
        annotated = re.sub(r'\b' + re.escape(entity_text) + r'\b', placeholder, annotated, flags=re.IGNORECASE)

    for placeholder, chip in placeholders.items():
        annotated = annotated.replace(placeholder, chip)

    legend = """
    <div class="legend">
      <div class="legend-item">
        <div class="legend-dot" style="background:#d1fae5;border:1px solid #6ee7b7;"></div>
        Matched in database
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#ffedd5;border:1px solid #fdba74;"></div>
        Not found in database
      </div>
    </div>
    """
    return f'{legend}<div class="annotated">{annotated}</div>'


def _build_results_html(results: list) -> str:
    matched = [r for r in results if r["matched"]]
    unmatched = [r for r in results if not r["matched"]]

    cards = []
    for r in matched:
        entity = r["entity"]["text"]
        label = r["entity"]["label"]
        best = r.get("highlighted_best_sentences") or r.get("highlighted_passage") or ""
        full = r.get("highlighted_passage") or ""
        topic = r.get("topic_title", "")
        url = r.get("source_url", "")
        url_display = url.replace("https://", "").rstrip("/")

        expand = ""
        if full and full != best:
            expand = f'<details><summary></summary><div class="full-passage">{full}</div></details>'

        cards.append(f"""
        <div class="card" id="{_card_id(entity)}">
          <div class="card-header">
            <span class="card-entity">{entity}</span>
            <span class="card-label">{label}</span>
          </div>
          <div class="card-topic"><span style="opacity:0.5;font-size:0.85em;font-weight:400;">Related MedlinePlus Health Topic : </span> <span>{topic}</span></div>
          <div class="card-passage">{best}</div>
          {expand}
          <div class="card-url"><a href="{url}" target="_blank">↗ {url_display}</a></div>
        </div>""")

    if unmatched:
        unmatched_cards = "\n".join(f"""
        <div class="card">
          <div class="card-header">
            <span class="card-entity">{r["entity"]["text"]}</span>
            <span class="card-label">{r["entity"]["label"]}</span>
          </div>
          <div class="no-match">No relevant information found in database.</div>
        </div>""" for r in unmatched)

        cards.append(f"""
        <details>
          <summary class="unmatched-toggle">↓ Show {len(unmatched)} unmatched entities</summary>
          {unmatched_cards}
        </details>""")

    return "\n".join(cards)


def analyze(text: str):
    if not text.strip():
        return "<p style='color:var(--body-text-color);opacity:0.4'>Enter text above to analyze.</p>"
    try:
        response = httpx.post(f"{API_BASE}/retrieve", json={"text": text}, timeout=60)
        response.raise_for_status()
        results = response.json()["results"]
    except Exception as e:
        return f"<p style='color:red'>API error: {e}</p>"

    annotation = _build_annotation_html(text, results)
    results_html = _build_results_html(results)
    return f"{annotation}<hr style='margin:24px 0;opacity:0.15'>{results_html}"


with gr.Blocks(css=CSS, title="Medical Anchor") as demo:
    gr.Markdown("# Medical Anchor\nGrounded medical entity extraction and retrieval from MedlinePlus.")

    text_input = gr.Textbox(
        label="Clinical text",
        value="Patient has asthma and hypertension, currently on claritin and lisinopril.",
        lines=6,
    )
    analyze_btn = gr.Button("Analyze", variant="primary")
    output = gr.HTML()

    analyze_btn.click(
        fn=analyze,
        inputs=text_input,
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()