"""Per-visitor state and helpers used by several tabs. State comes from st.session_state (one
per browser tab), never from module variables, which every visitor shares."""
import re

import pandas as pd
import streamlit as st
from openai import APIStatusError

import config
import rag as rag_module
from ui import i18n, resources
from ui.i18n import t

# Tabs (the label is t(key), in the interface language)
ASK, PRACTICE, BROWSE, COMPARE = "ask", "practice", "browse", "compare"
MODES = [ASK, PRACTICE, BROWSE, COMPARE]
AUTO = "auto"  # the model picker's "Auto" choice


# ---------- this visitor's state ----------
def rag():
    return st.session_state.rag


def corpus() -> str:
    """'demo', or a fingerprint of the uploaded files."""
    return st.session_state.corpus_key


def version() -> str:
    return st.session_state.version


def uploaded() -> bool:
    return corpus() != "demo"


def top_k() -> int:
    return st.session_state.get("k", config.TOP_K)


def lecture_filter():
    return st.session_state.get("lecture_filter") or None  # [] and None both mean all


def chosen_model():
    """The model picked in the sidebar, or None for Auto."""
    choice = st.session_state.get("model_choice", AUTO)
    return None if choice == AUTO else choice


def model_name(ref) -> str:
    return t("auto") if ref in (None, AUTO) else rag_module.model_label(ref)


# ---------- slides ----------
def slide_title(text: str, words: int = 9) -> str:
    """First words of a slide (its title, usually), without the footer or a leading all-caps
    section label such as "REVIEW AND REFERENCES". Only for indexes built before ingest.py
    read titles from the PDF."""
    head = re.split(r"\bLesson \d+\s*[—|/·:-]", text)[0]
    parts = head.split()
    caps = 0
    while caps < len(parts) and parts[caps].isupper() and len(re.sub(r"\W", "", parts[caps])) > 1:
        caps += 1
    if 2 <= caps < len(parts):
        parts = parts[caps:]
    for j, w in enumerate(parts):  # the title ends where a bullet or another label starts
        if j >= 2 and (w in {"•", "→"} or (w.isupper() and len(re.sub(r"\W", "", w)) >= 4)):
            parts = parts[:j]
            break
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def heading(chunk: dict, page: int | None = None) -> str:
    """A slide's title as printed on it (found by font size in ingest.py). page = one PDF page
    of a chunk that merged several; by default the chunk's main slide."""
    if page is not None and chunk.get("headings") and page in chunk["pages"]:
        title = chunk["headings"][chunk["pages"].index(page)]
    else:
        title = chunk.get("heading")
    return title or slide_title(chunk["text"])


def shorten(text: str, n: int = 48) -> str:
    return text if len(text) <= n else text[:n - 1].rsplit(" ", 1)[0] + "…"


def slide_caption(title: str, page: int, label) -> str:
    """'Intro · slide 59 (PDF page 56)': the number printed on the slide, plus where it is in
    the PDF when that differs (exported decks often skip hidden slides)."""
    if label is None:
        return t("caption_pdf_page", title=title, page=page)
    return t("caption_slide", title=title, label=label) + (
        t("caption_pdf_suffix", page=page) if label != page else "")


def image(pdf: str, page: int) -> bytes:
    return resources.slide_image(pdf, page, corpus(), rag().pdf_data.get(pdf))


def page_count(pdf: str) -> int:
    return resources.pdf_page_count(pdf, corpus(), rag().pdf_data.get(pdf))


def show_slide(h: dict):
    labels = h.get("labels") or [None] * len(h["pages"])
    for page, label in list(zip(h["pages"], labels))[-2:]:  # merged chunks: content is last
        st.image(image(h["pdf"], page), caption=slide_caption(h["title"], page, label))


def page_index(pdf: str) -> dict:
    """PDF page -> (printed label, chunk, slide title) for one lecture file."""
    out = {}
    for c in rag().chunks:
        if c["pdf"] == pdf:
            for page, label in zip(c["pages"], c.get("labels") or [None] * len(c["pages"])):
                out[page] = (label, c, heading(c, page))
    return out


def lecture_file(tag: str) -> str:
    return next(c["pdf"] for c in rag().chunks if c["lecture"] == tag)


# ---------- callbacks shared by tabs ----------
def thanks():
    st.toast(t("thanks"), icon=":material/favorite:")


def ask_lecture(tag: str):
    st.session_state.update(mode=ASK, lecture_filter=[tag])
    st.toast(t("searching_only_toast", tag=tag), icon=":material/filter_alt:")


def quiz_lecture(tag: str):
    st.session_state.update(mode=PRACTICE, quiz_lectures=[tag], quiz_autogen=True)


def api_error(e: APIStatusError):
    st.error(t("api_error", code=e.status_code), icon=":material/key_off:")


def show_answer_text(text: str):
    """An assistant answer, with citations as badges (and the busy message translated)."""
    st.markdown(t("busy") if text == config.BUSY_MESSAGE else rag_module.badge_citations(text))


# ---------- answers: how they were found, and their sources ----------
def chat_notes(messages: list) -> str:
    """The chat as Markdown study notes: each question, its answer, and the slides it cites."""
    lines = [t("notes_title"), ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"## {m['content']}", ""]
        else:
            lines += [m["content"], ""]
            cited = rag_module.cited_hits(m["content"], m.get("hits") or [])
            if cited:
                lines += [f"{t('notes_slides')}: "
                          + "; ".join(f"{h['cite']} ({heading(h)})" for h in cited), ""]
    return "\n".join(lines)


def trace_label(tr: dict) -> str:
    if tr.get("lectures"):  # a lecture named in the question, or the sidebar filter
        n = sum(c["lecture"] in tr["lectures"] for c in rag().chunks)
        scope = t("scope_lectures", n=n, lectures=", ".join(tr["lectures"]))
    else:
        scope = t("scope_all", n=tr["n_slides"])
    if tr.get("cached"):
        return t("trace_cached", scope=scope)
    label = t("trace_searched", scope=scope, s=tr["search_s"])
    if tr.get("model"):
        label += t("trace_answered", s=tr["answer_s"])
    return label


def trace_steps(tr: dict):
    """The steps that produced an answer, as a small timeline."""
    if tr.get("translation"):
        with st.status(t("step_translated"), type="step", state="complete"):
            st.caption(tr["translation"])
    scope = (", ".join(tr["lectures"]) + t("scope_named" if tr.get("named") else "scope_filter")
             ) if tr.get("lectures") else t("scope_all_lectures", n=tr["n_lectures"])
    with st.status(t("step_search", k=tr["k"]), type="step", state="complete"):
        st.caption(t("step_searched_scope", scope=scope))
    for ref, why in tr.get("failed") or []:
        with st.status(t("step_failed", model=model_name(ref)), type="step", state="error"):
            why = i18n.reason(why)
            st.caption(t("step_failed_reason", reason=why[:1].upper() + why[1:]))
    if tr.get("model"):
        with st.status(t("step_answer", model=model_name(tr["model"])), type="step",
                       state="complete"):
            st.caption(t("step_answer_detail", s=tr["answer_s"]))
    elif tr.get("refused_early"):
        with st.status(t("step_refused"), type="step", state="complete"):
            st.caption(t("step_refused_detail"))
    if tr.get("cached"):
        with st.status(t("step_cached"), type="step", state="complete"):
            st.caption(t("step_cached_detail"))


def slides_table(hits: list):
    table = pd.DataFrame([{t("col_slide"): h["cite"], t("col_title"): heading(h),
                           t("col_relevance"): h["score"], t("col_lecture"): h["title"]}
                          for h in hits])
    st.dataframe(table, hide_index=True, column_config={
        t("col_relevance"): st.column_config.ProgressColumn(
            min_value=0.6, max_value=1.0, format="%.2f", help=t("relevance_help"))})


def show_sources(answer: str, hits: list):
    cited = rag_module.cited_hits(answer, hits)
    if cited:
        st.markdown(f"**:material/location_on: {t('found_on')}** "
                    + " ".join(f":violet-badge[{h['cite']}]" for h in cited))
        with st.grid(2 if len(cited) > 1 else 1, gap="small"):
            for h in cited:
                show_slide(h)
    others = [h for h in hits if h not in cited]
    if others:
        label = t("other_slides") if cited else t("slides_searched")
        with st.expander(f"{label} ({len(others)})", icon=":material/search:"):
            slides_table(others)
