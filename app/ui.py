import re
import html
import os
from urllib.parse import urlparse
import httpx
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

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


def _safe_url(url: str) -> str:
    parsed = urlparse(url or "")
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url
    return ""


def _highlight_escaped_text(text: str, entity_text: str) -> str:
    escaped_text = html.escape(text or "")
    escaped_entity = html.escape(entity_text or "")
    if not escaped_text or not escaped_entity:
        return escaped_text
    pattern = re.compile(re.escape(escaped_entity), re.IGNORECASE)
    return pattern.sub(lambda m: f"<strong>{m.group(0)}</strong>", escaped_text)


def _build_annotation_html(input_text: str, results: list) -> str:
    matched_map = {r["entity"]["text"]: r["matched"] for r in results}
    label_map = {r["entity"]["text"]: r["entity"]["label"] for r in results}
    entities_sorted = sorted(matched_map.keys(), key=len, reverse=True)

    annotated = html.escape(input_text or "")
    placeholders = {}
    for i, entity_text in enumerate(entities_sorted):
        status = "matched" if matched_map[entity_text] else "unmatched"
        safe_entity = html.escape(entity_text)
        label = html.escape(label_map.get(entity_text, ""))
        if matched_map[entity_text]:
            chip = (
                f'<a class="entity {status}" href="#{_card_id(entity_text)}">'
                f'{safe_entity} <span class="label">{label}</span>'
                f'</a>'
            )
        else:
            chip = (
                f'<span class="entity {status}">'
                f'{safe_entity} <span class="label">{label}</span>'
                f'</span>'
            )
        placeholder = f"__ENTITY_{i}__"
        placeholders[placeholder] = chip
        escaped_entity = html.escape(entity_text)
        annotated = re.sub(r'\b' + re.escape(escaped_entity) + r'\b', placeholder, annotated, flags=re.IGNORECASE)

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
        entity_raw = r["entity"]["text"]
        entity = html.escape(entity_raw)
        label = html.escape(r["entity"]["label"])
        best_source = r.get("best_sentences") or r.get("passage") or ""
        full_source = r.get("passage") or ""
        best = _highlight_escaped_text(best_source, entity_raw)
        full = _highlight_escaped_text(full_source, entity_raw)
        topic = html.escape(r.get("topic_title", ""))
        safe_url = _safe_url(r.get("source_url", ""))
        url_display = html.escape(safe_url.replace("https://", "").replace("http://", "").rstrip("/"))

        expand = ""
        if full and full != best:
            expand = f'<details><summary></summary><div class="full-passage">{full}</div></details>'
        url_block = (
            f'<div class="card-url"><a href="{safe_url}" target="_blank" rel="noopener noreferrer">Source: {url_display}</a></div>'
            if safe_url else ""
        )

        cards.append(f"""
        <div class="card" id="{_card_id(entity_raw)}">
          <div class="card-header">
            <span class="card-entity">{entity}</span>
            <span class="card-label">{label}</span>
          </div>
          <div class="card-topic"><span style="opacity:0.5;font-size:0.85em;font-weight:400;">Related MedlinePlus Health Topic : </span> <span>{topic}</span></div>
          <div class="card-passage">{best}</div>
          {expand}
          {url_block}
        </div>""")

    if unmatched:
        unmatched_cards = "\n".join(f"""
        <div class="card">
          <div class="card-header">
            <span class="card-entity">{html.escape(r["entity"]["text"])}</span>
            <span class="card-label">{html.escape(r["entity"]["label"])}</span>
          </div>
          <div class="no-match">No relevant information found in database.</div>
        </div>""" for r in unmatched)

        cards.append(f"""
        <details>
          <summary class="unmatched-toggle">↓ Show {len(unmatched)} unmatched entities</summary>
          {unmatched_cards}
        </details>""")

    return "\n".join(cards)


def check_ready():
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            return (
                gr.update(value="✓ Models loaded — ready to analyze."),
                gr.update(interactive=True),
                gr.update(active=False),
            )
    except Exception:
        pass
    return (
        gr.update(value="⏳ Loading models, please wait..."),
        gr.update(interactive=False),
        gr.update(active=True),
    )


def analyze(text: str):
    if not text.strip():
        return "<p style='color:var(--body-text-color);opacity:0.4'>Enter text above to analyze.</p>"
    try:
        response = httpx.post(f"{API_BASE}/retrieve", json={"text": text}, timeout=60)
        response.raise_for_status()
        results = response.json()["results"]
    except Exception:
        return "<p style='color:red'>API error: request failed.</p>"

    annotation = _build_annotation_html(text, results)
    results_html = _build_results_html(results)
    return f"{annotation}<hr style='margin:24px 0;opacity:0.15'>{results_html}"


with gr.Blocks(title="Medical Anchor") as demo:
    gr.Markdown("# Medical Anchor\nGrounded medical entity extraction and retrieval from MedlinePlus.")

    text_input = gr.Textbox(
        label="Clinical text",
        value="Patient has asthma and hypertension, currently on claritin and lisinopril.",
        lines=6,
    )
    analyze_btn = gr.Button("Analyze", variant="primary", interactive=False)
    status = gr.Markdown("⏳ Loading models, please wait...")
    output = gr.HTML()

    analyze_btn.click(fn=analyze, inputs=text_input, outputs=output)
    timer = gr.Timer(3, active=True)
    timer.tick(fn=check_ready, outputs=[status, analyze_btn, timer])

if __name__ == "__main__":
    demo.launch(css=CSS, server_name="0.0.0.0", server_port=7860)
