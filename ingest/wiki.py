"""Full Wikipedia articles (`wikimedia/wikipedia`, 20231101.en) turned into corpus documents.

Each article's whole lead section is replaced by the exact 2017 HotpotQA paragraph, so gold sentence
labels stay valid and the modern lead cannot contradict them (DATASET.md §1). Trailing boilerplate
("See also", "References", ...) is dropped: it is link lists, not prose.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.document import Section, StructuredText, render
from app.ingestion.text import Span, content_hash

WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"

BOILERPLATE_HEADINGS = {
    "see also", "references", "external links", "notes", "further reading", "bibliography",
    "sources", "citations", "footnotes", "notes and references", "references and notes",
    "explanatory notes", "general and cited sources", "works cited", "literature",
}

_HEADING_MAX_CHARS = 80
_HEADING_MAX_WORDS = 12
_NOT_A_HEADING_END = re.compile(r"[.!?:;,]$")


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    source: str
    structured: StructuredText
    sha256: str
    lead_replaced: bool  # the modern article had its own lead section, which we discarded
    gold_sentence_spans: list[Span] = field(default_factory=list)  # span of 2017 sentence i


def download(target_dir: Path) -> list[Path]:
    """Fetch the English parquet shards (~11.6 GB, once). A plain local_dir rather than the HF cache:
    on Windows without symlinks the cache would keep a second copy of every shard."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        WIKI_REPO,
        repo_type="dataset",
        allow_patterns=f"{WIKI_CONFIG}/*.parquet",
        local_dir=target_dir,
    )
    return sorted(Path(target_dir, WIKI_CONFIG).glob("*.parquet"))


def read_titles(files: list[Path]) -> set[str]:
    import pyarrow.parquet as pq

    titles: set[str] = set()
    for f in files:
        titles.update(pq.read_table(f, columns=["title"]).column("title").to_pylist())
    return titles


def read_articles(files: list[Path], wanted: set[str]) -> dict[str, tuple[str, str]]:
    """title -> (url, text) for the wanted titles, streaming row batches to bound memory."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    value_set = pa.array(sorted(wanted), type=pa.string())
    found: dict[str, tuple[str, str]] = {}
    for f in files:
        reader = pq.ParquetFile(f)
        for batch in reader.iter_batches(columns=["title", "url", "text"], batch_size=20_000):
            hits = batch.filter(pc.is_in(batch.column("title"), value_set=value_set))
            for title, url, text in zip(*(hits.column(c).to_pylist() for c in ("title", "url", "text"))):
                found.setdefault(title, (url, text))
    return found


def parse_sections(text: str) -> list[Section]:
    """Split plain-text Wikipedia into sections. Headings are short standalone lines without
    closing punctuation; list items are lines starting with a space and stay inside paragraphs."""
    sections: list[Section] = []
    heading: str | None = None
    paragraphs: list[str] = []
    current: list[str] = []

    def end_paragraph() -> None:
        if current:
            paragraph = "\n".join(line.strip() for line in current).strip()
            if len(paragraph) > 1:  # drops stray "." left where formulas were stripped
                paragraphs.append(paragraph)
            current.clear()

    for line in text.split("\n"):
        if not line.strip():
            end_paragraph()
        elif not current and _is_heading(line):
            end_paragraph()
            sections.append(Section(heading, paragraphs))
            if line.strip().lower() in BOILERPLATE_HEADINGS:
                return [s for s in sections if s.paragraphs]
            heading, paragraphs = line.strip(), []
        else:
            current.append(line)
    end_paragraph()
    sections.append(Section(heading, paragraphs))
    return [s for s in sections if s.paragraphs]


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return (
        not line.startswith(" ")  # list item
        and len(stripped) <= _HEADING_MAX_CHARS
        and len(stripped.split()) <= _HEADING_MAX_WORDS
        and not _NOT_A_HEADING_END.search(stripped)
    )


def build_doc(
    doc_id: str, title: str, url: str, wiki_text: str, hotpot_sentences: list[str]
) -> CorpusDoc:
    """Lead = the 2017 HotpotQA paragraph; body = the 2023 article minus its own lead section."""
    sentences = [s.strip() for s in hotpot_sentences if s.strip()]
    if not sentences:
        raise ValueError(f"no HotpotQA paragraph for {title!r}")
    lead = " ".join(sentences)
    sections = parse_sections(wiki_text)
    had_modern_lead = bool(sections) and sections[0].heading is None
    body = [s for s in sections if s.heading is not None]
    structured = render([Section(None, [lead])] + body)

    # The lead paragraph is the first thing rendered, so its sentences start at offset 0.
    spans: list[Span] = []
    pos = 0
    for sentence in sentences:
        spans.append((pos, pos + len(sentence)))
        pos += len(sentence) + 1  # the joining space
    return CorpusDoc(
        doc_id=doc_id,
        title=title,
        source=url,
        structured=structured,
        sha256=content_hash(structured.text),
        lead_replaced=had_modern_lead,
        gold_sentence_spans=spans,
    )
