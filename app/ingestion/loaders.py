"""File loaders (BACKEND.md §3.1): PDF, HTML, Markdown and plain text -> the same Section structure.

Each loader recovers what structure its format really has, because the recursive chunker splits on it:
HTML and Markdown have explicit headings; plain text gets the same "short standalone line" heading rule
the Wikipedia parser uses; PDF has no reliable structure, so it becomes paragraphs only.
"""

import re
from dataclasses import dataclass, field
from pathlib import PurePath

from app.ingestion.document import Section
from app.ingestion.text import normalize

MAX_BYTES = 10 * 1024 * 1024
FORMATS = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "markdown", ".markdown": "markdown",
           ".txt": "text"}  # fmt: skip


class LoaderError(ValueError):
    """The file cannot be ingested: unsupported, too large, unreadable or empty."""


@dataclass
class LoadedDocument:
    title: str
    format: str
    filename: str
    sections: list[Section]
    metadata: dict = field(default_factory=dict)


def load(data: bytes, filename: str) -> LoadedDocument:
    fmt = FORMATS.get(PurePath(filename).suffix.lower())
    if fmt is None:
        raise LoaderError(f"unsupported file type {PurePath(filename).suffix or '(none)'}; "
                          f"supported: {', '.join(sorted(FORMATS))}")  # fmt: skip
    if len(data) > MAX_BYTES:
        raise LoaderError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    title, sections, metadata = {"pdf": _pdf, "html": _html, "markdown": _markdown, "text": _text}[fmt](data)
    sections = [s for s in sections if any(p.strip() for p in s.paragraphs)]
    if not sections:
        raise LoaderError("no text could be extracted (a scanned PDF needs OCR first)")
    return LoadedDocument(title or PurePath(filename).stem, fmt, filename, sections, metadata)


# --- plain text ---------------------------------------------------------------------------------

_HEADING_END = re.compile(r"[.,;:!?)\]\"']$")


def _decode(data: bytes) -> str:
    try:
        return normalize(data.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return normalize(data.decode("cp1252", errors="replace"))


def _is_heading(block: str) -> bool:
    return "\n" not in block and len(block) <= 80 and len(block.split()) <= 12 and not _HEADING_END.search(block)


def plain_sections(text: str) -> list[Section]:
    """Blank-line separated blocks; a short one-line block without closing punctuation is a heading."""
    sections, heading, paragraphs = [], None, []
    for block in (b.strip() for b in re.split(r"\n\s*\n", text)):
        if not block:
            continue
        if _is_heading(block):
            sections.append(Section(heading, paragraphs))
            heading, paragraphs = block, []
        else:
            paragraphs.append(re.sub(r"[ \t]*\n[ \t]*", "\n", block))
    sections.append(Section(heading, paragraphs))
    return sections


def _text(data: bytes):
    text = _decode(data)
    first = text.split("\n", 1)[0].strip()
    return (first if _is_heading(first) else None), plain_sections(text), {}


# --- markdown -----------------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # images -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links -> link text
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),  # bold
    (re.compile(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])"), r"\2"),  # italics
    (re.compile(r"`([^`]+)`"), r"\1"),  # inline code
]


def _md_inline(text: str) -> str:
    for pattern, repl in _MD_INLINE:
        text = pattern.sub(repl, text)
    return text


def _markdown(data: bytes):
    text = _decode(data)
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # YAML front matter
    sections, heading, paragraphs, current, title = [], None, [], [], None
    in_fence = False

    def end_paragraph():
        if current:
            paragraphs.append("\n".join(current).strip())
            current.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        match = None if in_fence else _MD_HEADING.match(line)
        if match:
            end_paragraph()
            sections.append(Section(heading, paragraphs))
            heading, paragraphs = _md_inline(match.group(2)), []
            if title is None and len(match.group(1)) == 1:
                title = heading
        elif not line.strip() and not in_fence:
            end_paragraph()
        else:
            current.append(line if in_fence else _md_inline(re.sub(r"^\s{0,3}>\s?", "", line)))
    end_paragraph()
    sections.append(Section(heading, paragraphs))
    return title, sections, {}


# --- HTML ---------------------------------------------------------------------------------------

_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th", "dd", "dt"]


def _html(data: bytes):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    sections, heading, paragraphs = [], None, []
    root = soup.body or soup
    for el in root.find_all(_BLOCKS):
        if el.find_parent(_BLOCKS[6:]):  # nested block (e.g. <p> inside <li>): the parent has its text
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name.startswith("h") and len(el.name) == 2:
            sections.append(Section(heading, paragraphs))
            heading, paragraphs = text, []
            title = title or (text if el.name == "h1" else None)
        else:
            paragraphs.append(text)
    if not any(p for s in sections for p in s.paragraphs) and not paragraphs:
        paragraphs = [p for p in plain_sections(root.get_text("\n"))[0].paragraphs]  # no block tags at all
    sections.append(Section(heading, paragraphs))
    return title, sections, {}


# --- PDF ----------------------------------------------------------------------------------------


def _pdf(data: bytes):
    import io

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as e:
        raise LoaderError(f"unreadable PDF: {e}") from e
    meta = reader.metadata or {}
    title = (meta.get("/Title") or "").strip() or None
    paragraphs = [p for page in pages for p in _pdf_paragraphs(normalize(page))]
    return title, [Section(None, paragraphs)], {"pages": len(pages)}


def _pdf_paragraphs(page: str) -> list[str]:
    """PDF text comes as visual lines. A line ending a sentence that is noticeably shorter than the
    page's long lines ends a paragraph; other line breaks are wrapping and become spaces (rejoining
    hyphenated words)."""
    lines = [line.strip() for line in page.split("\n")]
    width = max((len(line) for line in lines), default=0)
    paragraphs, current = [], []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if current and current[-1].endswith("-") and line[:1].islower():
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)
        if line.endswith((".", "!", "?", ":")) and len(line) < 0.8 * width:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs
