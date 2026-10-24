import pytest

from app.ingestion.loaders import LoaderError, load


def make_pdf(lines: list[str], title: str = "Test PDF") -> bytes:
    """A minimal one-page PDF with Helvetica text lines, built by hand so the test needs no PDF writer."""
    ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    for line in lines:
        ops.append(f"({line}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Title (" + title.encode() + b") >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 6 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_markdown_headings_become_sections_and_syntax_is_stripped():
    md = b"---\nlayout: post\n---\n# Jaws\n\nA **1975** film by [Spielberg](http://x).\n\n## Plot\n\nA shark.\n"
    doc = load(md, "jaws.md")
    assert doc.title == "Jaws" and doc.format == "markdown"
    assert [(s.heading, s.paragraphs) for s in doc.sections] == [
        ("Jaws", ["A 1975 film by Spielberg."]), ("Plot", ["A shark."])]  # fmt: skip


def test_html_drops_scripts_and_navigation_and_keeps_heading_structure():
    html = b"""<html><head><title>Jaws</title><script>evil()</script></head><body>
    <nav>Home | About</nav><h1>Jaws</h1><p>A 1975 film.</p><h2>Cast</h2><ul><li>Roy <b>Scheider</b></li></ul>
    <footer>(c) site</footer></body></html>"""
    doc = load(html, "jaws.html")
    assert doc.title == "Jaws"
    assert [(s.heading, s.paragraphs) for s in doc.sections] == [("Jaws", ["A 1975 film."]), ("Cast", ["Roy Scheider"])]


def test_plain_text_uses_the_short_line_heading_rule():
    txt = "Jaws\n\nJaws is a 1975 film.\nIt was a hit.\n\nProduction\n\nFilming began in 1974.".encode()
    doc = load(txt, "notes.txt")
    assert doc.title == "Jaws"
    assert [s.heading for s in doc.sections] == ["Jaws", "Production"]
    assert doc.sections[0].paragraphs == ["Jaws is a 1975 film.\nIt was a hit."]


def test_pdf_text_and_title_are_extracted():
    doc = load(make_pdf(["Jaws is a 1975 American thriller film directed by Steven", "Spielberg.", "",
                         "It was a box office hit."]), "jaws.pdf")  # fmt: skip
    assert doc.title == "Test PDF" and doc.format == "pdf" and doc.metadata == {"pages": 1}
    text = " ".join(p for s in doc.sections for p in s.paragraphs)
    assert "directed by Steven Spielberg." in text and "box office hit" in text


@pytest.mark.parametrize("data,name,message", [
    (b"x", "a.docx", "unsupported"),
    (b"", "empty.txt", "no text"),
    (b"not a pdf", "broken.pdf", "unreadable PDF"),
])  # fmt: skip
def test_bad_files_raise_a_loader_error(data, name, message):
    with pytest.raises(LoaderError, match=message):
        load(data, name)
