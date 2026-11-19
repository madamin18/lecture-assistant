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


def cite_tag(lec: str, pages: list, labels: list | None = None) -> str:
    """'L01 slide 59' when the slides print their own numbers, otherwise 'L01 p.56' (PDF page)."""
    if labels and all(n is not None for n in labels):
        nums, word = labels, "slide "
    else:
        nums, word = pages, "p."
    return f"{lec} {word}{nums[0]}" if len(nums) == 1 else f"{lec} {word}{nums[0]}-{nums[-1]}"


def combine(a: dict, b: dict) -> dict:
    pages = a["pages"] + b["pages"]
    labels = a.get("labels", [None] * len(a["pages"])) + b.get("labels", [None] * len(b["pages"]))
    headings = ((a.get("headings") or [a.get("heading", "")] * len(a["pages"]))
                + (b.get("headings") or [b.get("heading", "")] * len(b["pages"])))
    main = b if len(b["text"]) >= len(a["text"]) else a  # the content slide, not the short one
    return {**a, "pages": pages, "labels": labels, "headings": headings,
            "text": (a["text"] + " " + b["text"]).strip(),
            "heading": main.get("heading") or a.get("heading") or b.get("heading") or "",
            "cite": cite_tag(a["lecture"], pages, labels)}


def slide_heading(page) -> str:
    """The slide's title: the largest text in the top 40% of the page. Stops at the first gap,
    so a big-font statement further down the slide isn't added to it."""
    height = page.rect.height
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                text = s["text"].strip()
                x0, y0, _, y1 = s["bbox"]
                if text and y0 < 0.4 * height and re.search(r"[^\W\d_]", text):  # has letters
                    spans.append((s["size"], y0, x0, y1, text))
    if not spans:
        return ""
    biggest = max(s[0] for s in spans)
    big = sorted((s for s in spans if s[0] >= biggest - 1.5), key=lambda s: (round(s[1] / 4), s[2]))
    parts, bottom = [], None
    for size, y0, _, y1, text in big:
        if bottom is not None and y0 - bottom > 0.5 * size:
            break
        parts.append(text)
        bottom = y1 if bottom is None else max(bottom, y1)
    return clean(" ".join(parts))


LECTURE_WORDS = {"lesson", "lecture", "week", "chapter", "unit", "part", "session"}


def printed_number(page) -> int | None:
    """The slide number printed in the page's footer, if any. Takes the right-most number,
    so 'Lesson 01 ... 59' gives 59; '12 / 40' gives 12. (Headers are ignored: titles like
    'Step 1', 'Step 2' would look like slide numbers.)"""
    height = page.rect.height
    words = page.get_text("words")
    for top, bottom in ((0.85, 1.01),):  # bottom 15% of the page
        band = sorted((w for w in words if top * height <= w[1] < bottom * height),
                      key=lambda w: w[0])
        m = re.search(r"(\d{1,3})\s*/\s*\d{1,3}$", " ".join(w[4] for w in band))
        if m:
            return int(m.group(1))
        # skip "Lesson 7" / "Week 3" inside the footer text: on a slide without its own number,
        # "Lesson 7 / Testing" would otherwise read as slide 7. A number that ends the footer
        # is still the slide number ("Industry special lecture 12" is slide 12).
        digits = [w for j, w in enumerate(band) if re.fullmatch(r"\d{1,3}", w[4])
                  and not (0 < j < len(band) - 1 and band[j - 1][4].lower() in LECTURE_WORDS)]
        if digits:
            return int(max(digits, key=lambda w: w[0])[4])
    return None


def slide_labels(doc) -> list:
    """Printed slide numbers for every page, or all None if they don't look reliable.
    Exported decks often skip numbers (hidden slides), so PDF page 56 can be slide 59."""
    nums = [printed_number(p) for p in doc]
    found = [n for n in nums if n is not None]
    reliable = (len(found) >= 3 and len(found) >= 0.7 * len(nums) and found[0] >= 1
                and all(a < b for a, b in zip(found, found[1:])))
    if not reliable:
        return [None] * len(nums)
    # Fill a missing number when the neighbours leave only one possibility, e.g. an unnumbered
    # title page before slide 2 is slide 1. Otherwise leave it unknown (cited by PDF page).
    filled = list(nums)
    for i, n in enumerate(nums):
        if n is not None:
            continue
        before = nums[i - 1] if i > 0 else 0
        after = nums[i + 1] if i + 1 < len(nums) else None
        if before is not None and after is not None and after - before == 2:
            filled[i] = before + 1
    return filled


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
        labels = slide_labels(doc)
        for page_no, page in enumerate(doc, start=1):
            label = labels[page_no - 1]
            head = slide_heading(page)
            slides.append({
                "lecture": lec, "title": title, "pdf": ref,
                "pages": [page_no],    # position in the PDF (used to render the image)
                "labels": [label],     # number printed on the slide, None if unknown
                "text": clean(page.get_text()),
                "heading": head,       # the slide's own title, shown in the website
                "headings": [head],    # one per page (several after merge_short)
                "cite": cite_tag(lec, [page_no], [label]),
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
