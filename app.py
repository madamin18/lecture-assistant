"""Step 3: the website.  Run:  python -m streamlit run app.py

Four tabs (each in ui/):
  Ask              chat with the slides; each answer shows how it was found and the slides it
                   came from, with follow-up suggestions
  Practice exam    generated exam-style questions with instant feedback and a running score
  Browse lectures  lecture cards and a slide viewer, with shortcuts to ask about a slide or
                   get quizzed on a lecture
  Compare models   the same question to 2-3 AI models side by side, with votes

Visitors can upload their own lecture PDFs (kept in memory for their browser tab only, never
saved on the server). With no uploads, the site uses the course slides in index/ (built by
ingest.py). This file sets up the page, the sidebar and the tabs.
"""
import hashlib
import importlib
import time
from pathlib import Path

import streamlit as st

import answer_cache
import config
import ingest
import rag as rag_module
import ui.ask
import ui.browse
import ui.common
import ui.compare
import ui.practice
import ui.resources

# Reloaded in this order (each module after the ones it imports).
MODULES = (config, answer_cache, ingest, rag_module, ui.resources, ui.common,
           ui.ask, ui.practice, ui.browse, ui.compare)
DATA_FILES = ("index/chunks.json", "index/answer_cache.json")


def code_version() -> str:
    """Fingerprint of the code, settings, demo index and saved answers."""
    h = hashlib.sha1()
    here = Path(__file__).parent
    for path in [Path(m.__file__) for m in MODULES] + [here / f for f in DATA_FILES]:
        try:
            h.update(path.read_bytes())
        except OSError:  # no demo index: the site runs upload-only
            pass
    return h.hexdigest()[:12]


def load_code() -> str:
    """A deploy that only changes the modules (not app.py) doesn't restart the server, and Python
    keeps running the old ones (rag.py keeps reading the old config). Whenever the files differ
    from what is loaded, reload them in dependency order. Returns an id that changes on every
    reload; cached objects take it as an argument, so new code always gets new objects (also
    when a change is reverted)."""
    version = code_version()
    if getattr(rag_module, "LOADED_VERSION", None) != version:
        for module in MODULES:
            importlib.reload(module)
        rag_module.LOADED_VERSION = version
        rag_module.LOAD_ID = f"{version}-{time.time_ns()}"
    return rag_module.LOAD_ID


VERSION = load_code()
c, resources = ui.common, ui.resources

st.set_page_config(page_title="Lecture Notes Assistant", page_icon=":material/school:")

if not any(p["key"] for p in config.PROVIDERS.values()):
    st.error("No API key found. Add GEMINI_API_KEY (or GROQ_API_KEY, OPENROUTER_API_KEY) to "
             "`.env` (local) or to the app's Secrets on Streamlit Cloud (Manage app -> Settings "
             "-> Secrets).")
    st.stop()


# ---------- uploads ----------
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
            pages += ingest.page_count(name, pdf)
        except Exception:
            return None, [("error", f"**{name}** couldn't be opened as a PDF.")]
    if pages > config.MAX_UPLOAD_PAGES:
        return None, [("error", f"These files have {pages} pages in total. The limit is "
                                f"{config.MAX_UPLOAD_PAGES}; please upload fewer lectures.")]

    with st.spinner(f"Reading {pages} slides… (first time takes about a minute)"):
        chunks, emb, skipped = ingest.build_index(list(data.items()), resources.load_embedder())

    notes = [("warning", f"**{name}** has no readable text (probably scanned images), so it "
                         "can't be searched.") for name in skipped]
    if not chunks:
        return None, notes
    return rag_module.SlideRAG(chunks=chunks, emb=emb, pdf_data=data,
                               embedder=resources.load_embedder(),
                               client=resources.load_clients(VERSION)), notes


def corpus_key(files) -> str:
    if not files:
        return "demo"
    return "-".join(hashlib.sha1(f.getvalue()).hexdigest()[:12] for f in files)


with st.sidebar:
    st.header("Your slides", anchor=False)
    files = st.file_uploader(
        "Upload lecture PDFs", type="pdf", accept_multiple_files=True,
        help="Only you can search the files you upload. They stay in memory for this "
             "browser tab and are never saved on the server.")

# Re-index only when the set of uploaded files (or the app's code) changes;
# a new set starts a new chat and quiz.
key = corpus_key(files)
if st.session_state.get("corpus_key") != key or st.session_state.get("version") != VERSION:
    if files:
        rag, notes = index_uploads(files)
        if rag:
            st.toast(f"Ready: {len(rag.chunks)} slides from {len(files)} file(s)",
                     icon=":material/check_circle:")
    else:
        rag, notes = resources.load_demo(VERSION), []
    st.session_state.update(corpus_key=key, version=VERSION, rag=rag, notes=notes,
                            messages=[], quiz=None, comparison=None, lecture_filter=[],
                            quiz_lectures=[])
for name, default in (("messages", []), ("quiz", None), ("comparison", None),
                      ("lecture_filter", []), ("quiz_lectures", []), ("mode", c.ASK)):
    st.session_state.setdefault(name, default)
rag = st.session_state.rag

# Jumps requested from the slide viewer dialog (applied before any widget is drawn).
goto = st.session_state.pop("goto", None)
if goto:
    where, payload = goto
    if where == "ask":
        st.session_state.update(mode=c.ASK, pending_question=payload)
    elif where == "quiz":
        st.session_state.update(mode=c.PRACTICE, quiz_lectures=[payload], quiz_autogen=True)


# ---------- sidebar: search and model settings ----------
models = resources.available_models(VERSION)
with st.sidebar:
    if rag is not None and len(rag.lectures) > 1:
        st.pills("Search only these lectures", [t for t, _ in rag.lectures],
                 selection_mode="multi", key="lecture_filter",
                 help="Leave empty to search every lecture. Naming a lecture in your question "
                      "(\"in Lesson 3\", \"3강\") also works.")
    st.slider("Slides read per answer", 3, 12, config.TOP_K, key="k",
              help="How many of the best-matching slides the assistant reads before answering.")
    choices = [c.AUTO] + models
    if st.session_state.get("model_choice") not in choices:  # e.g. a free model was retired
        st.session_state.model_choice = c.AUTO
    st.selectbox("AI model", choices, key="model_choice",
                 format_func=lambda m: "Auto (recommended)" if m == c.AUTO
                 else rag_module.model_label(m),
                 help="Auto starts with the most reliable model and moves on to the next one, "
                      "or another provider, when a model is busy or out of free quota. Pick a "
                      "model to try it; the Compare models tab puts several side by side.")
    with st.container(horizontal=True, gap="small"):
        if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.messages = []
        st.download_button("Save chat", lambda m=list(st.session_state.messages): c.chat_notes(m),
                           file_name="lecture_chat.md", mime="text/markdown",
                           icon=":material/download:", width="stretch", on_click="ignore",
                           disabled=not st.session_state.messages,
                           help="Download this chat as study notes (Markdown), with the "
                                "slides each answer came from.")
    st.caption("Answers and practice questions come only from the slides and can still be "
               "wrong. Check the slide shown with each one before relying on it.")


# ---------- header ----------
st.title("Lecture Notes Assistant", anchor=False)
st.caption("Ask about your lectures, practise for the exam, or flip through the slides. "
           "Every answer shows the slide it came from.")

for level, text in st.session_state.notes:
    getattr(st, level)(text)

if rag is None:
    if not files:
        st.info("Upload one or more lecture PDFs in the sidebar to get started.",
                icon=":material/upload_file:")
    st.stop()

n_lectures = len(rag.lectures)
providers = {rag_module.split_ref(m)[0] for m in models}
st.markdown(
    f":violet-badge[:material/menu_book: {n_lectures} {'uploaded' if files else 'course'} "
    f"lecture{'s' * (n_lectures > 1)}] "
    f":blue-badge[:material/slideshow: {len(rag.chunks)} slides] "
    ":green-badge[:material/translate: English · 한국어] "
    f":gray-badge[:material/smart_toy: {len(models)} AI models"
    f"{f' from {len(providers)} providers' if len(providers) > 1 else ''}] "
    ":orange-badge[:material/verified: Answers cite their slides]")
if not files:
    st.caption("Demo: the lectures of *Design of Generative AI Application Systems*. "
               "Upload your own PDFs in the sidebar to use those instead.")

mode = st.segmented_control("Mode", [c.ASK, c.PRACTICE, c.BROWSE, c.COMPARE], required=True,
                            key="mode", label_visibility="collapsed", width="stretch")

{c.PRACTICE: ui.practice.page, c.BROWSE: ui.browse.page,
 c.COMPARE: ui.compare.page}.get(mode, ui.ask.page)()
