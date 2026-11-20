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
import os
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
import ui.games
import ui.i18n
import ui.practice
import ui.resources

# Reloaded in this order (each module after the ones it imports).
MODULES = (config, answer_cache, ingest, rag_module, ui.i18n, ui.resources, ui.common,
           ui.ask, ui.practice, ui.browse, ui.compare, ui.games)
DATA_FILES = ("index/chunks.json", "index/answer_cache.json")


def sync_secrets():
    """Streamlit Cloud turns the app's Secrets into environment variables only when the server
    starts, so a key added later (e.g. GROQ_API_KEY) would need a reboot. st.secrets always has
    the current ones: copy them over on every run."""
    try:
        secrets = {k: v for k, v in st.secrets.items() if isinstance(v, str)}
    except Exception:  # noqa: BLE001  (no secrets file: running locally with .env)
        return
    for name, value in secrets.items():
        if os.environ.get(name) != value:
            os.environ[name] = value


def code_version() -> str:
    """Fingerprint of the code, settings (including which API keys are set), demo index and
    saved answers."""
    h = hashlib.sha1()
    here = Path(__file__).parent
    files = ([Path(m.__file__) for m in MODULES] + [here / f for f in DATA_FILES]
             + sorted(here.glob("ui/game_assets/*")))  # the games' JS, CSS and word pairs
    for path in files:
        try:
            h.update(path.read_bytes())
        except OSError:  # no demo index: the site runs upload-only
            pass
    for name in sorted(os.environ):  # a new or changed key rebuilds the AI clients
        if name.endswith(("_API_KEY", "_MODELS")) or name.startswith("LLM_"):
            h.update(f"{name}={os.environ[name]}".encode())
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


sync_secrets()
VERSION = load_code()
c, resources, t = ui.common, ui.resources, ui.i18n.t

st.session_state.setdefault("ui_lang", "en")
st.set_page_config(page_title=t("app_title"), page_icon=":material/school:")

# Interface language, at the very top of the sidebar.
with st.sidebar:
    st.segmented_control(t("ui_language"), list(ui.i18n.LANGUAGES), key="ui_lang",
                         format_func=ui.i18n.LANGUAGES.get, required=True,
                         label_visibility="collapsed", width="stretch")

if not any(p["key"] for p in config.PROVIDERS.values()):
    st.error(t("no_key"))
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
            return None, [("error", "upload_bad_pdf", {"name": name})]
    if pages > config.MAX_UPLOAD_PAGES:
        return None, [("error", "upload_too_many",
                       {"pages": pages, "limit": config.MAX_UPLOAD_PAGES})]

    with st.spinner(t("upload_reading", pages=pages)):
        chunks, emb, skipped = ingest.build_index(list(data.items()), resources.load_embedder())

    # messages are kept as (level, text key, values), so they follow the interface language
    notes = [("warning", "upload_no_text", {"name": name}) for name in skipped]
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
    st.header(t("sb_your_slides"), anchor=False)
    files = st.file_uploader(t("sb_upload"), type="pdf", accept_multiple_files=True,
                             help=t("sb_upload_help"))

# Re-index only when the set of uploaded files (or the app's code) changes;
# a new set starts a new chat and quiz.
key = corpus_key(files)
if st.session_state.get("corpus_key") != key or st.session_state.get("version") != VERSION:
    if files:
        rag, notes = index_uploads(files)
        if rag:
            st.toast(t("upload_ready", slides=len(rag.chunks), files=len(files)),
                     icon=":material/check_circle:")
    else:
        rag, notes = resources.load_demo(VERSION), []
    st.session_state.update(corpus_key=key, version=VERSION, rag=rag, notes=notes,
                            messages=[], quiz=None, comparison=None, lecture_filter=[],
                            quiz_lectures=[])
for name, default in (("messages", []), ("quiz", None), ("comparison", None),
                      ("lecture_filter", []), ("quiz_lectures", []), ("mode", c.ASK)):
    st.session_state.setdefault(name, default)
if st.session_state.mode not in c.MODES:  # e.g. a tab name saved by an older version
    st.session_state.mode = c.ASK
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
        st.pills(t("sb_filter"), [tag for tag, _ in rag.lectures], selection_mode="multi",
                 key="lecture_filter", help=t("sb_filter_help"))
    st.slider(t("sb_k"), 3, 12, config.TOP_K, key="k", help=t("sb_k_help"))
    choices = [c.AUTO] + models
    if st.session_state.get("model_choice") not in choices:  # e.g. a free model was retired
        st.session_state.model_choice = c.AUTO
    # One picker per interface language (a picker keeps showing the old language's text for the
    # selected item); the choice itself lives in "model_choice", shared by both.
    picker = f"model_picker_{ui.i18n.lang()}"
    st.session_state[picker] = st.session_state.model_choice
    st.selectbox(t("sb_model"), choices, key=picker,
                 format_func=lambda m: t("auto_recommended") if m == c.AUTO
                 else rag_module.model_label(m), help=t("sb_model_help"),
                 on_change=lambda: st.session_state.update(model_choice=st.session_state[picker]))
    with st.container(horizontal=True, gap="small"):
        if st.button(t("sb_clear"), icon=":material/delete_sweep:", width="stretch"):
            st.session_state.messages = []
        notes_text = c.chat_notes(st.session_state.messages)  # in the current language
        st.download_button(t("sb_save"), lambda text=notes_text: text,
                           file_name="lecture_chat.md", mime="text/markdown",
                           icon=":material/download:", width="stretch", on_click="ignore",
                           disabled=not st.session_state.messages, help=t("sb_save_help"))
    st.caption(t("sb_disclaimer"))
    if st.button(t("games_btn"), icon=":material/sports_esports:", type="primary",
                 width="stretch"):
        ui.games.open_games()


# ---------- header ----------
st.title(t("app_title"), anchor=False)
st.caption(t("app_subtitle"))

for level, key_, values in st.session_state.notes:
    getattr(st, level)(t(key_, **values))

if rag is None:
    if not files:
        st.info(t("upload_start"), icon=":material/upload_file:")
    st.stop()

n_lectures = len(rag.lectures)
providers = {rag_module.split_ref(m)[0] for m in models}
st.markdown(
    f":violet-badge[:material/menu_book: {t('badge_lectures', n=n_lectures, uploaded=bool(files))}] "
    f":blue-badge[:material/slideshow: {t('badge_slides', n=len(rag.chunks))}] "
    ":green-badge[:material/translate: English · 한국어 · Русский] "
    f":gray-badge[:material/smart_toy: {t('badge_models', n=len(models), p=len(providers))}] "
    f":orange-badge[:material/verified: {t('badge_cites')}]")
if not files:
    st.caption(t("demo_caption"))

mode = st.segmented_control("Mode", c.MODES, required=True, key="mode", format_func=t,
                            label_visibility="collapsed", width="stretch")

{c.PRACTICE: ui.practice.page, c.BROWSE: ui.browse.page,
 c.COMPARE: ui.compare.page}.get(mode, ui.ask.page)()
