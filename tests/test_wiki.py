from ingest.wiki import build_doc, parse_sections

# Shaped like real wikimedia/wikipedia text: blank-line paragraphs, headings on their own line
# (sometimes with a trailing space and a single newline), list items indented by one space.
ARTICLE = (
    "Albedo is the fraction of sunlight reflected. It is measured from 0 to 1.\n\n"
    "A second lead paragraph restates a 2023 fact.\n\n"
    "Terrestrial albedo\n\n"
    "Snow has a high albedo. Forests are darker.\n\n"
    "History \n"
    "The term was introduced by Lambert in 1760.\n\n"
    ".\n\n"
    "See also\n\n"
    " Cool roof\n Daisyworld\n\n"
    "References\n\n"
    "Climatology\n1760s neologisms"
)


def test_parse_sections_finds_lead_headings_and_strips_boilerplate():
    sections = parse_sections(ARTICLE)
    assert [s.heading for s in sections] == [None, "Terrestrial albedo", "History"]
    assert len(sections[0].paragraphs) == 2
    assert sections[2].paragraphs == ["The term was introduced by Lambert in 1760."]
    assert all("Cool roof" not in p for s in sections for p in s.paragraphs)


def test_build_doc_replaces_whole_lead_and_keeps_gold_spans_exact():
    hotpot = ["Albedo is the 2017 measure of reflection.", " It ranges from 0 to 1."]
    doc = build_doc("id-1", "Albedo", "https://en.wikipedia.org/wiki/Albedo", ARTICLE, hotpot)

    text = doc.structured.text
    assert text.startswith("Albedo is the 2017 measure of reflection. It ranges from 0 to 1.")
    assert "restates a 2023 fact" not in text  # the entire modern lead is gone, not just para 1
    assert "Snow has a high albedo." in text
    assert doc.lead_replaced
    assert [text[s:e] for s, e in doc.gold_sentence_spans] == [
        "Albedo is the 2017 measure of reflection.",
        "It ranges from 0 to 1.",
    ]


def test_build_doc_without_modern_lead_is_flagged():
    doc = build_doc("id-2", "T", "u", "History\n\nOnly a body.", ["The 2017 lead."])
    assert not doc.lead_replaced
    assert doc.structured.text == "The 2017 lead.\n\nHistory\n\nOnly a body."
