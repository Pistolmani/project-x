"""The normalized structure every loader produces, rendered to one string with known spans.

Rendering everything to a single string and keeping character spans for each heading and paragraph
means every chunk (from either strategy) is an exact substring with known offsets. Evaluation can then
check "is this gold sentence fully inside this chunk?" by comparing offsets, with no fuzzy matching.
"""

from dataclasses import dataclass, field

from app.ingestion.text import Span

PARAGRAPH_SEP = "\n\n"


@dataclass(frozen=True)
class Section:
    heading: str | None  # None for the lead / untitled opening section
    paragraphs: list[str]


@dataclass(frozen=True)
class SectionSpans:
    heading: Span | None
    paragraphs: list[Span]


@dataclass
class StructuredText:
    text: str
    sections: list[SectionSpans] = field(default_factory=list)


def render(sections: list[Section]) -> StructuredText:
    """Join headings and paragraphs with blank lines, recording where each one lands."""
    parts: list[str] = []
    spans: list[SectionSpans] = []
    pos = 0

    def add(piece: str) -> Span:
        nonlocal pos
        if parts:
            parts.append(PARAGRAPH_SEP)
            pos += len(PARAGRAPH_SEP)
        parts.append(piece)
        span = (pos, pos + len(piece))
        pos += len(piece)
        return span

    for section in sections:
        paragraphs = [p.strip() for p in section.paragraphs if p.strip()]
        if not paragraphs:
            continue  # a heading with no body adds nothing searchable
        heading_span = add(section.heading.strip()) if section.heading else None
        spans.append(SectionSpans(heading_span, [add(p) for p in paragraphs]))

    return StructuredText("".join(parts), spans)
