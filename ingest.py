"""Step 1: read slide PDFs -> one chunk per slide -> embeddings.

Used two ways:
  python ingest.py        builds the demo index from data/slides/ into index/
  build_index(...)        called by app.py on PDFs a visitor uploads (kept in memory only)
"""
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz
import numpy as np

import config


def lecture_info(filename: str):
    """'L03_RAG_basics.pdf' -> ('L03', 'RAG basics').  'Week 3 notes.pdf' -> ('Week_3_notes', 'Week 3 notes')."""
    stem = Path(filename).stem
    m = re.match(r"(L\d+)[_\- ]*(.*)", stem, re.IGNORECASE)
    if m:
        lec = m.group(1).upper()
        title = m.group(2).replace("_", " ").strip() or lec
        return lec, title
    safe = re.sub(r"\W+", "_", stem).strip("_") or "doc"
    return safe, stem


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cite_tag(lec: str, pages: list) -> str:
    return f"{lec} p.{pages[0]}" if len(pages) == 1 else f"{lec} p.{pages[0]}-{pages[-1]}"


def combine(a: dict, b: dict) -> dict:
    pages = a["pages"] + b["pages"]
    return {**a, "pages": pages, "text": (a["text"] + " " + b["text"]).strip(),
            "cite": cite_tag(a["lecture"], pages)}


def open_pdf(ref: str, data: bytes | None = None):
    """Open from disk (data=None) or from uploaded bytes."""
    return fitz.open(stream=data, filetype="pdf") if data is not None else fitz.open(ref)


def page_count(ref: str, data: bytes | None = None) -> int:
    with open_pdf(ref, data) as doc:
        return doc.page_count


def extract_slides(ref: str, data: bytes | None = None, lec: str | None = None) -> list:
    """ref = path on disk, or the uploaded file's name when data (the PDF bytes) is given."""
    auto_lec, title = lecture_info(ref)
    lec = lec or auto_lec
    slides = []
    with open_pdf(ref, data) as doc:
        for page_no, page in enumerate(doc, start=1):
            slides.append({
                "lecture": lec, "title": title, "pdf": ref,
                "pages": [page_no], "text": clean(page.get_text()),
                "cite": cite_tag(lec, [page_no]),
            })
    return slides


def merge_short(slides: list) -> list:
    """Title slides / diagram-only slides have little text; attach them to the next slide."""
    merged, carry = [], None
    for s in slides:
        if carry:
            s, carry = combine(carry, s), None
        if len(s["text"]) < config.MIN_CHARS:
            carry = s
            continue
        merged.append(s)
    if carry:  # short last slide -> attach to previous one
        if merged:
            merged[-1] = combine(merged[-1], carry)
        elif carry["text"]:
            merged.append(carry)
    return merged


def embed_chunks(chunks: list, model) -> np.ndarray:
    # e5 models expect a "passage: " prefix; adding the lecture title gives each slide context.
    texts = [f"passage: {c['lecture']} {c['title']}. {c['text']}" for c in chunks]
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return model.encode(texts, normalize_embeddings=True, batch_size=32).astype("float32")


def build_index(pdfs: list, model):
    """pdfs = [(ref, data_or_None), ...].  Returns (chunks, embeddings, skipped_refs).

    skipped_refs lists PDFs with no extractable text (usually scanned images)."""
    chunks, skipped, used = [], [], Counter()
    for ref, data in pdfs:
        lec, _ = lecture_info(ref)
        used[lec] += 1
        if used[lec] > 1:  # two uploads named alike -> keep citation tags unique
            lec = f"{lec}_{used[lec]}"
        slides = merge_short(extract_slides(ref, data, lec))
        if not slides:
            skipped.append(ref)
        chunks.extend(slides)
    return chunks, embed_chunks(chunks, model), skipped


def main():
    from sentence_transformers import SentenceTransformer

    pdfs = sorted(Path(config.SLIDES_DIR).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.SLIDES_DIR}/")

    model = SentenceTransformer(config.EMBED_MODEL)
    # as_posix(): "data/slides/x.pdf" also works on Linux (Streamlit Cloud)
    chunks, emb, skipped = build_index([(p.as_posix(), None) for p in pdfs], model)
    for pdf, n in Counter(c["pdf"] for c in chunks).items():
        print(f"{Path(pdf).name}: {n} chunks")
    for ref in skipped:
        print(f"{Path(ref).name}: no text found (scanned PDF?) - skipped")

    out = Path(config.INDEX_DIR)
    out.mkdir(exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {out}/")


if __name__ == "__main__":
    main()
