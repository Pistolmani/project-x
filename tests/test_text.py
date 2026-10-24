from app.ingestion.document import Section, render
from app.ingestion.text import content_hash, sentence_spans


def test_content_hash_ignores_whitespace_only_differences():
    assert content_hash("A  b\n\nc ") == content_hash("A b c")
    assert content_hash("A b c") != content_hash("A b d")


def test_sentence_spans_split_on_terminal_punctuation():
    text = 'He left. "Why?" she asked. It was 1990! Then quiet'
    assert [text[s:e] for s, e in sentence_spans(text)] == [
        "He left.",
        '"Why?" she asked.',
        "It was 1990!",
        "Then quiet",
    ]


def test_sentence_spans_respect_bounds_and_return_absolute_offsets():
    text = "Intro. One here. Two here. Outro."
    start = text.index("One")
    end = text.index(" Outro")
    assert [text[s:e] for s, e in sentence_spans(text, start, end)] == ["One here.", "Two here."]


def test_render_records_exact_spans_and_drops_empty_sections():
    doc = render(
        [
            Section(None, ["Lead para."]),
            Section("Empty", ["  "]),
            Section("History", ["First.", "Second."]),
        ]
    )
    assert doc.text == "Lead para.\n\nHistory\n\nFirst.\n\nSecond."
    lead, history = doc.sections
    assert lead.heading is None
    assert [doc.text[s:e] for s, e in lead.paragraphs] == ["Lead para."]
    assert doc.text[slice(*history.heading)] == "History"
    assert [doc.text[s:e] for s, e in history.paragraphs] == ["First.", "Second."]
