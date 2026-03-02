from app.ui import _build_annotation_html, _build_results_html, _safe_url


def test_annotation_escapes_script_payload():
    text = "<script>alert(1)</script> patient has asthma"
    results = [
        {"entity": {"text": "asthma", "label": "PROBLEM"}, "matched": True},
    ]

    html_out = _build_annotation_html(text, results)

    assert "<script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out


def test_results_escape_entity_and_topic_fields():
    results = [
        {
            "matched": True,
            "entity": {"text": '<img src=x onerror=alert(1)>', "label": "PROBLEM"},
            "topic_title": "<b>Asthma</b>",
            "best_sentences": "Asthma can affect breathing.",
            "passage": "Asthma can affect breathing.",
            "source_url": "https://medlineplus.gov/asthma.html",
        }
    ]

    html_out = _build_results_html(results)

    assert "<img" not in html_out
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_out
    assert "<b>Asthma</b>" not in html_out
    assert "&lt;b&gt;Asthma&lt;/b&gt;" in html_out


def test_rejects_javascript_urls():
    assert _safe_url("javascript:alert(1)") == ""
    assert _safe_url("data:text/html;base64,xxx") == ""
    assert _safe_url("https://example.com") == "https://example.com"
