"""Objects shared by every visitor, loaded once per server (Streamlit caches). Functions that take
`version` are rebuilt when the code or the demo index changes (see load_code in app.py)."""
import threading

import streamlit as st

import answer_cache
import config
import ingest
import rag as rag_module


@st.cache_resource(show_spinner="Loading the search model…")
def load_embedder():
    from sentence_transformers import SentenceTransformer  # slow import: once per server
    return SentenceTransformer(config.EMBED_MODEL)


@st.cache_resource
def load_clients(version: str) -> dict:
    """{provider: API client} for every provider with a key."""
    return rag_module.make_clients()


@st.cache_resource(show_spinner=False)
def load_demo(version: str):
    try:
        return rag_module.SlideRAG(embedder=load_embedder(), client=load_clients(version))
    except FileNotFoundError:  # no index/ folder: site runs upload-only
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(version: str) -> list:
    """Every model the site's keys can use ("groq:qwen/qwen3.8-27b", ...), main provider first.
    Fetched from the providers at most once an hour."""
    clients = load_clients(version)
    return [m for p in config.available_providers() if p in clients
            for m in rag_module.list_models(p, clients[p])]


# ---------- answers already given ----------
RUNTIME_CACHE_SIZE = 300  # answers kept in memory; the oldest are dropped first


@st.cache_resource
def runtime_answers(version: str) -> dict:
    """First questions already answered on this server, shared by all visitors."""
    return {}


def remember_answer(version: str, ckey: str, entry: dict):
    cache = runtime_answers(version)
    cache[ckey] = entry
    while len(cache) > RUNTIME_CACHE_SIZE:
        try:
            cache.pop(next(iter(cache)))
        except (StopIteration, RuntimeError, KeyError):  # another visitor changed it meanwhile
            break


@st.cache_resource(show_spinner=False)
def saved_answers(version: str) -> dict:
    """Answers prepared by warm_cache.py for the demo slides."""
    return answer_cache.load_file()


# ---------- lecture summaries ----------
@st.cache_resource(show_spinner=False)
def saved_summaries(version: str) -> dict:
    """Summaries of the demo lectures prepared by warm_cache.py, {"L03|en": entry}."""
    return answer_cache.load_summaries()


@st.cache_resource
def site_summaries(version: str) -> dict:
    """Summaries of the demo lectures written on this server, shared by all visitors.
    Summaries of uploaded slides are never put here (they stay in the visitor's own tab)."""
    return {}


# ---------- slide images ----------
@st.cache_data(max_entries=400, show_spinner=False)
def slide_image(pdf_ref: str, page: int, doc_key: str, _data: bytes | None) -> bytes:
    # doc_key identifies the PDF's content; _data (leading underscore = not hashed) is the upload.
    return rag_module.render_page(pdf_ref, page, _data)


@st.cache_data(show_spinner=False)
def pdf_page_count(pdf_ref: str, doc_key: str, _data: bytes | None) -> int:
    return ingest.page_count(pdf_ref, _data)


# ---------- model votes (Compare tab) ----------
@st.cache_resource
def vote_board() -> dict:
    """Votes from everyone on this server since it started: {model: counts}. Not tied to the
    code version, so a deploy doesn't wipe them (a server restart does)."""
    return {"lock": threading.Lock(), "models": {}}


def record_vote(results: list, verdict):
    """results = the compared answers; verdict = index of the better answer, "tie" or "bad"."""
    board = vote_board()
    with board["lock"]:
        for i, r in enumerate(results):
            row = board["models"].setdefault(r["model"], {
                "votes": 0, "wins": 0, "ties": 0, "bad": 0, "seconds": 0.0, "failures": 0})
            if r["text"] is None:
                row["failures"] += 1
                continue
            row["votes"] += 1
            row["seconds"] += r["seconds"]
            row["wins"] += verdict == i
            row["ties"] += verdict == "tie"
            row["bad"] += verdict == "bad"
