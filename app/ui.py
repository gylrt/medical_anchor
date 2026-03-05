import re
import html
import os
from urllib.parse import urlparse
import httpx
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DAILYMED_PREVIEW_SENTENCES = 2

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


def _first_n_sentences(text: str, n: int = DAILYMED_PREVIEW_SENTENCES) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return raw
    return " ".join(parts[:n])


def _clean_dailymed_passage(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    kept = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("title:") or low.startswith("section:"):
            continue
        kept.append(ln)
    return " ".join(kept).strip()


def _build_annotation_html(input_text: str, results: list) -> str:
    text = input_text or ""
    spans = []
    for r in results:
        entity = r.get("entity", {})
        entity_text = entity.get("text", "")
        start = entity.get("start")
        end = entity.get("end")
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        if not (0 <= start < end <= len(text)):
            continue

        status = "matched" if r.get("matched") else "unmatched"
        label = html.escape(entity.get("label", ""))
        display = html.escape(text[start:end])
        if r.get("matched"):
            chip = (
                f'<a class="entity {status}" href="#{_card_id(entity_text)}">'
                f'{display} <span class="label">{label}</span>'
                f'</a>'
            )
        else:
            chip = (
                f'<span class="entity {status}">'
                f'{display} <span class="label">{label}</span>'
                f'</span>'
            )
        spans.append((start, end, chip))

    if not spans:
        annotated = html.escape(text)
    else:
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        parts = []
        cursor = 0
        for start, end, chip in spans:
            if start < cursor:
                continue
            parts.append(html.escape(text[cursor:start]))
            parts.append(chip)
            cursor = end
        parts.append(html.escape(text[cursor:]))
        annotated = "".join(parts)

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
        label_raw = (r["entity"].get("label") or "").strip().lower()
        label = html.escape(r["entity"]["label"])
        is_dailymed = label_raw == "treatment"
        if is_dailymed:
            raw_chunk = (r.get("passage") or "").strip()
            cleaned = _clean_dailymed_passage(raw_chunk)
            snippet = _first_n_sentences(cleaned, DAILYMED_PREVIEW_SENTENCES)
            best_source = f"INDICATIONS AND USAGE: {snippet}" if snippet else ""
            full_source = raw_chunk
        else:
            best_source = r.get("best_sentences") or r.get("passage") or ""
            full_source = r.get("passage") or ""
        best = _highlight_escaped_text(best_source, entity_raw)
        full = _highlight_escaped_text(full_source, entity_raw)
        topic = html.escape(r.get("topic_title", ""))
        generic_name = html.escape((r.get("generic_name") or "").strip())
        safe_url = _safe_url(r.get("source_url", ""))
        url_display = html.escape(safe_url.replace("https://", "").replace("http://", "").rstrip("/"))

        expand = ""
        if is_dailymed and full:
            expand = f'<details><summary></summary><div class="full-passage">{full}</div></details>'
        elif full and full != best:
            expand = f'<details><summary></summary><div class="full-passage">{full}</div></details>'
        url_block = (
            f'<div class="card-url"><a href="{safe_url}" target="_blank" rel="noopener noreferrer">Source: {url_display}</a></div>'
            if safe_url else ""
        )
        if is_dailymed:
            generic = (
                ' <span style="opacity:0.5;font-size:0.85em;font-weight:400;">'
                f'| Generic name: </span><span>{generic_name}</span>'
                if generic_name else ""
            )
            topic_block = (
                '<div class="card-topic">'
                '<span style="opacity:0.5;font-size:0.85em;font-weight:400;">Related DailyMed product: </span> '
                f'<span>{topic}</span>{generic}'
                '</div>'
            )
        else:
            topic_block = (
                '<div class="card-topic"><span style="opacity:0.5;font-size:0.85em;font-weight:400;">'
                'Related MedlinePlus Health Topic : </span> '
                f'<span>{topic}</span></div>'
            )

        cards.append(f"""
        <div class="card" id="{_card_id(entity_raw)}">
          <div class="card-header">
            <span class="card-entity">{entity}</span>
            <span class="card-label">{label}</span>
          </div>
          {topic_block}
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
        value=(
            "44-year-old female with Sjögren’s syndrome, ADHD, insomnia, anxiety, and depression reports a burning skin in the "
            "evenings. The skin becomes very hot and painful before visible redness appears, especially where skin is in contact "
            "with clothing. Current medications include Seroquel 150 mg, lamotrigine 200 mg, Adderall 25 mg, omeprazole, "
            "cevimeline 30 mg, and oxybutynin 10 mg. She was evaluated by primary care and dermatology, and lupus was discussed "
            "as a possible contributor but remains unconfirmed."
        ),
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
