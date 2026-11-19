"""Reading slide PDFs: titles, printed slide numbers, merging, citation tags."""
import pymupdf as fitz

from ingest import (build_index, cite_tag, lecture_info, merge_short, printed_number,
                    slide_heading, slide_labels)
from conftest import FakeEmbedder

BODY = "This slide explains the idea in enough words to count as real content for a chunk."


def make_pdf(pages):
    """pages = [(title, body, footer number or None)] -> PDF bytes (16:9 slides)."""
    doc = fitz.open()
    for title, body, number in pages:
        page = doc.new_page(width=960, height=540)
        if title:
            page.insert_text((40, 60), title, fontsize=28)
        if body:
            page.insert_text((40, 200), body, fontsize=12)
        page.insert_text((40, 520), "Lesson 7 / Testing", fontsize=8)
        if number is not None:
            page.insert_text((900, 520), str(number), fontsize=8)
    return doc.tobytes()


def test_lecture_info():
    assert lecture_info("data/slides/L03_RAG_basics.pdf") == ("L03", "RAG basics")
    assert lecture_info("Week 3 notes.pdf") == ("Week_3_notes", "Week 3 notes")


def test_cite_tag_prefers_printed_numbers():
    assert cite_tag("L01", [56], [59]) == "L01 slide 59"
    assert cite_tag("L01", [5, 6], [None, 7]) == "L01 p.5-6"
    assert cite_tag("L01", [5, 6], [8, 9]) == "L01 slide 8-9"


def test_heading_and_footer_number():
    doc = fitz.open(stream=make_pdf([("Chunk overlap", BODY, 12)]), filetype="pdf")
    assert slide_heading(doc[0]) == "Chunk overlap"
    assert printed_number(doc[0]) == 12  # the right-most footer number, not "Lesson 7"


def test_footer_number_after_lecture_word():
    # L07's real footer: "Lesson 07 | Industry special lecture 2" is slide 2
    doc = fitz.open()
    page = doc.new_page(width=960, height=540)
    page.insert_text((40, 520), "Lesson 07 | Industry special lecture 2", fontsize=8)
    assert printed_number(page) == 2


def test_slide_labels_keep_skipped_numbers_and_fill_one_gap():
    # an unnumbered first page before slide 2 is slide 1; hidden slide 4 was not exported
    doc = fitz.open(stream=make_pdf([("Cover", BODY, None), ("A", BODY, 2), ("B", BODY, 3),
                                     ("C", BODY, 5), ("D", BODY, 6)]), filetype="pdf")
    assert slide_labels(doc) == [1, 2, 3, 5, 6]


def test_slide_labels_unreliable_numbers_are_ignored():
    doc = fitz.open(stream=make_pdf([("A", BODY, 3), ("B", BODY, 2), ("C", BODY, 1)]),
                    filetype="pdf")
    assert slide_labels(doc) == [None, None, None]


def test_merge_short_attaches_title_slide_to_next():
    slides = [
        {"lecture": "L01", "pages": [1], "labels": [1], "text": "Intro", "heading": "Intro",
         "headings": ["Intro"], "cite": "L01 slide 1"},
        {"lecture": "L01", "pages": [2], "labels": [2], "text": BODY, "heading": "Main idea",
         "headings": ["Main idea"], "cite": "L01 slide 2"},
    ]
    [merged] = merge_short(slides)
    assert merged["cite"] == "L01 slide 1-2"
    assert merged["heading"] == "Main idea"  # the content slide's title
    assert merged["headings"] == ["Intro", "Main idea"]


def test_build_index_from_uploaded_bytes():
    pdf = make_pdf([("Chunk overlap", BODY, 1), ("Metadata", BODY, 2), ("Filters", BODY, 3)])
    chunks, emb, skipped = build_index([("L07_Testing.pdf", pdf)], FakeEmbedder())
    assert [c["cite"] for c in chunks] == ["L07 slide 1", "L07 slide 2", "L07 slide 3"]
    assert [c["heading"] for c in chunks] == ["Chunk overlap", "Metadata", "Filters"]
    assert emb.shape == (3, FakeEmbedder.dim) and skipped == []


def test_build_index_reports_pdfs_without_text():
    doc = fitz.open()
    doc.new_page()
    chunks, emb, skipped = build_index([("scan.pdf", doc.tobytes())], FakeEmbedder())
    assert chunks == [] and skipped == ["scan.pdf"] and emb.shape[0] == 0
