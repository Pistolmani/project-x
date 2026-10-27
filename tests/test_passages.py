from ui.passages import as_plain_markdown, excerpt, group_citations, label

ARTICLE = ("Jacob the Liar (East Germany)\n\nJaws (June 20), directed by Steven Spielberg\n"
           "Keetje Tippel, directed by Paul Verhoeven. The Killer Elite, directed by Sam Peckinpah.")


def cite(sid, start, end, doc="d1", idx=None, ce=None):
    return {"source_id": sid, "doc_id": doc, "title": "1975 in film", "text": ARTICLE[start:end],
            "chunk_index": idx, "start_char": start, "end_char": end, "ce_score": ce}


def test_overlapping_chunks_merge_into_the_exact_article_text():
    # cited later chunk first, as the agent did ([S1] = chunk #9, [S2] = chunk #8)
    [p] = group_citations([cite("S1", 40, len(ARTICLE), idx=9, ce=6.13), cite("S2", 0, 75, idx=8, ce=5.2)])
    assert p.text == ARTICLE and p.text.count("Jaws (June 20)") == 1
    assert p.source_ids == ["S1", "S2"] and p.chunk_indices == [8, 9] and p.best_ce == 6.13
    assert label(p) == "[S1, S2] 1975 in film · chunks #8–9 · best CE score 6.13"


def test_touching_chunks_merge_and_separate_ones_do_not():
    [p] = group_citations([cite("S1", 0, 30), cite("S2", 30, 60)])
    assert p.text == ARTICLE[0:60]
    apart = group_citations([cite("S1", 0, 20), cite("S2", 100, 120), cite("S3", 10, 40, doc="d2")])
    assert [q.source_ids for q in apart] == [["S1"], ["S2"], ["S3"]]


def test_contained_chunk_and_citations_without_offsets():
    [p] = group_citations([cite("S1", 0, 80), cite("S2", 10, 40)])
    assert p.text == ARTICLE[0:80] and p.source_ids == ["S1", "S2"]
    no_offsets = [{"source_id": "S1", "doc_id": "d1", "title": "T", "text": "a"},
                  {"source_id": "S2", "doc_id": "d1", "title": "T", "text": "b"}]
    assert len(group_citations(no_offsets)) == 2
    assert label(group_citations(no_offsets)[0]) == "[S1] T"


def test_excerpt_marks_only_cut_edges():
    assert excerpt("Jaws is a film.", 0) == "Jaws is a film."
    assert excerpt("Lügner) (East Germany)", 8544) == "…Lügner) (East Germany)"
    assert excerpt("directed by Robert", None) == "directed by Robert…"


def test_plain_markdown_keeps_lines_and_literal_characters():
    md = as_plain_markdown("# Not a heading\n*not bold* 1975.\n\nnext paragraph")
    assert md.startswith("\\# Not a heading  \n\\*not bold\\* 1975\\.")
    assert md.count("  \n") == 3 and md.endswith("next paragraph")
