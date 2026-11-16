"""Step 3: the website.  Run:  streamlit run app.py

Visitors upload their own lecture PDFs (kept in memory for their browser tab only,
never saved on the server) and ask questions. Each answer shows the slides it came from.
With no uploads, the site searches the demo slides in index/ (built by ingest.py).
"""
import hashlib
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

import config
from ingest import build_index, page_count
from rag import SlideRAG, cited_hits, make_client, render_page

st.set_page_config(page_title="Lecture Notes Assistant", page_icon="📚")

if not config.LLM_API_KEY:
    st.error("No API key found. Add GEMINI_API_KEY to `.env` (local) or to the app's "
             "Secrets on Streamlit Cloud (Manage app -> Settings -> Secrets).")
    st.stop()


# ---------- shared by all visitors (loaded once per server) ----------
@st.cache_resource(show_spinner="Loading the search model…")
def load_embedder():
    return SentenceTransformer(config.EMBED_MODEL)


@st.cache_resource
def load_client():
    return make_client()


@st.cache_resource(show_spinner=False)
def load_demo():
    try:
        return SlideRAG(embedder=load_embedder(), client=load_client())
    except FileNotFoundError:  # no index/ folder: site runs upload-only
        return None


@st.cache_data(max_entries=300, show_spinner=False)
def slide_image(pdf_ref: str, page: int, doc_key: str, _data: bytes | None) -> bytes:
    # doc_key identifies the PDF's content; _data (leading underscore = not hashed) is the upload.
    return render_page(pdf_ref, page, _data)


# ---------- per visitor ----------
def index_uploads(files):
    """Returns (SlideRAG or None, list of messages to show)."""
    data = {}
    for f in files:
        name = f.name
        n = 2
        while name in data:  # same file name uploaded twice
            name = f"{Path(f.name).stem} ({n}).pdf"
            n += 1
        data[name] = f.getvalue()

    pages = 0
    for name, pdf in data.items():
        try:
            pages += page_count(name, pdf)
        except Exception:
            return None, [("error", f"**{name}** couldn't be opened as a PDF.")]
    if pages > config.MAX_UPLOAD_PAGES:
        return None, [("error", f"These files have {pages} pages in total. The limit is "
                                f"{config.MAX_UPLOAD_PAGES}; please upload fewer lectures.")]

    with st.spinner(f"Reading {pages} slides… (first time takes about a minute)"):
        chunks, emb, skipped = build_index(list(data.items()), load_embedder())

    notes = [("warning", f"**{name}** has no readable text (probably scanned images), so it "
                         "can't be searched.") for name in skipped]
    if not chunks:
        return None, notes
    rag = SlideRAG(chunks=chunks, emb=emb, pdf_data=data,
                   embedder=load_embedder(), client=load_client())
    return rag, notes


def corpus_key(files) -> str:
    if not files:
        return "demo"
    return "-".join(hashlib.sha1(f.getvalue()).hexdigest()[:12] for f in files)


# ---------- sidebar ----------
with st.sidebar:
    st.header("Your lecture slides")
    files = st.file_uploader(
        "Upload lecture PDFs", type="pdf", accept_multiple_files=True,
        help="Only you can search the files you upload. They stay in memory for this "
             "browser tab and are never saved on the server.")
    st.markdown("---")
    k = st.slider("Slides to search", 3, 10, config.TOP_K)
    show_images = st.toggle("Show previews of other searched slides", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
    st.markdown("---")
    st.caption("Answers come only from the slides and can still be wrong. "
               "Check the slide shown under each answer before relying on it.")

# Re-index only when the set of uploaded files changes; a new set starts a new chat.
key = corpus_key(files)
if st.session_state.get("corpus_key") != key:
    if files:
        rag, notes = index_uploads(files)
    else:
        rag, notes = load_demo(), []
    st.session_state.update(corpus_key=key, rag=rag, notes=notes, messages=[])
rag = st.session_state.rag

# ---------- main page ----------
st.title("Lecture Notes Assistant")
st.caption("Upload lecture slides, ask a question, and see the exact slide the answer came from.")

for level, text in st.session_state.notes:
    getattr(st, level)(text)

if rag is None:
    if not files:
        st.info("Upload one or more lecture PDFs in the sidebar to get started.")
    st.stop()

if not files:
    titles = sorted({c["title"] for c in rag.chunks})
    st.info(f"**Demo mode:** searching the sample lecture *{', '.join(titles)}*. "
            "Upload your own PDFs in the sidebar to search those instead.")
else:
    n_files = len({c["pdf"] for c in rag.chunks})
    st.success(f"Searching {n_files} uploaded file(s), {len(rag.chunks)} slides.")


def show_slide(h):
    data = rag.pdf_data.get(h["pdf"])
    for page in h["pages"][-2:]:  # merged chunks: the content is on the last pages
        st.image(slide_image(h["pdf"], page, key, data),
                 caption=f"{h['title']} · page {page}")


def show_sources(answer, hits):
    cited = cited_hits(answer, hits)
    if cited:
        st.markdown("**📍 Found on:** " + ", ".join(f"`{h['cite']}`" for h in cited))
        for h in cited:
            show_slide(h)
    others = [h for h in hits if h not in cited]
    if others:
        label = "Other slides searched" if cited else "Slides searched"
        with st.expander(f"{label} ({len(others)})"):
            for h in others:
                st.markdown(f"**[{h['cite']}]** {h['title']} — relevance {h['score']:.2f}")
                if show_images:
                    show_slide(h)
                else:
                    st.caption(h["text"][:300] + ("…" if len(h["text"]) > 300 else ""))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["content"], m["hits"])

if question := st.chat_input("Ask a question about the slides"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the slides…"):
            answer, hits = rag.answer(question, k)
        st.markdown(answer)
        show_sources(answer, hits)
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
