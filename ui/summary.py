"""Summary tab: a one-page study summary of a lecture (overview, key points, key terms), written
by the AI from that lecture's slides only, with every point citing the slides it comes from.
Uploading PDFs opens this tab and summarizes the first uploaded lecture straight away."""
import re
import time

import streamlit as st
from openai import APIStatusError

import answer_cache
import rag as rag_module
from ui import browse
from ui import common as c
from ui import i18n, resources
from ui.i18n import t


def _key(tag: str) -> str:
    """Summaries are kept per slide set, lecture, interface language and model."""
    return f"{c.corpus()}|{tag}|{i18n.lang()}|{c.chosen_model() or c.AUTO}"


def find(tag: str) -> dict | None:
    """A summary written before: in this browser tab, else (course slides only) on this server
    or prepared by warm_cache.py (those were written by Auto, so only when Auto is chosen)."""
    found = st.session_state.summaries.get(_key(tag))
    if found or c.uploaded():
        return found
    found = resources.site_summaries(c.version()).get(_key(tag))
    if found is None and c.chosen_model() is None:
        found = resources.saved_summaries(c.version()).get(
            answer_cache.summary_key(tag, i18n.lang()))
    return {**found, "cached": True} if found else None


def citations(tag: str, text: str) -> dict:
    """How many slides of the lecture the summary cites, and how many of its citations don't
    match one (see rag.citation_facts)."""
    return rag_module.citation_facts(text, [ch for ch in c.rag().chunks if ch["lecture"] == tag])


def write(tag: str) -> dict | None:
    """Stream a new summary onto the page. Returns it, or None if no model could write it."""
    info, text, box = {}, "", st.empty()
    t0 = time.time()
    try:
        with st.spinner(t("sum_writing", tag=tag)):
            for piece in c.rag().summarize(tag, i18n.lang(), info, model=c.chosen_model()):
                text += piece
                box.markdown(rag_module.badge_citations(text) + " ▌")
    except APIStatusError as e:
        c.api_error(e)
        return None
    if info.get("model") is None or "cut off" in text:  # every model busy, or stopped midway
        box.warning(t("busy"), icon=":material/hourglass_top:")
        return None
    entry = {"text": text, "model": info["model"], "seconds": round(time.time() - t0, 1),
             "slides": info["slides"], "trimmed": info["trimmed"]}
    st.session_state.summaries[_key(tag)] = entry
    # Course slides are public: share the summary with other visitors, unless a citation is off
    # (once in 27 test summaries, Flash-Lite wrote [L14 slide 14] for [L06 slide 14]).
    if not c.uploaded() and not citations(tag, text)["made_up"]:
        resources.site_summaries(c.version())[_key(tag)] = entry
    return entry


def note(tag: str, entry: dict) -> str:
    """Who wrote the summary, from how many slides, and how many different slides it cites."""
    facts = citations(tag, entry["text"])
    text = t("sum_saved" if entry.get("cached") else "sum_written",
             model=c.model_name(entry["model"]), s=entry["seconds"],
             slides=t("n_slides", n=entry["slides"]), cited=t("n_slides", n=facts["cited"]))
    if entry.get("trimmed"):
        text += " " + t("sum_trimmed")
    if facts["made_up"]:
        text += " " + t("sum_made_up", n=facts["made_up"])
    return text


def summary_file(tag: str, title: str, entry: dict) -> str:
    """The summary as a Markdown file, citations kept as [L03 slide 14]."""
    return f"# {lecture_name(tag, title)}\n\n{entry['text'].strip()}\n\n---\n\n*{note(tag, entry)}*\n"


def show(tag: str, title: str, entry: dict):
    with st.container(border=True):
        c.show_answer_text(entry["text"])
    st.caption(note(tag, entry))
    file_name = re.sub(r"[^\w-]+", "_", tag) + "_summary.md"  # an upload's tag may have spaces
    with st.container(horizontal=True, gap="small"):
        st.download_button(t("sum_download"), summary_file(tag, title, entry),
                           file_name=file_name, mime="text/markdown",
                           icon=":material/download:", on_click="ignore")
        st.button(t("quiz_lecture"), icon=":material/quiz:", on_click=c.quiz_lecture,
                  args=(tag,))
        st.button(t("sum_ask"), icon=":material/forum:", on_click=c.ask_lecture, args=(tag,))
        if st.button(t("view_slides"), icon=":material/slideshow:"):
            browse.VIEWERS[i18n.lang()](tag)
        if citations(tag, entry["text"])["made_up"]:
            st.button(t("sum_retry"), icon=":material/refresh:",
                      on_click=lambda: st.session_state.update(summary_retry=tag))


def lecture_name(tag: str, title: str) -> str:
    """'L03 · RAG I Ingestion and Indexing'; just the title when the tag only repeats it (an
    upload without a lecture number: 'Week 1 slides', not 'Week_1_slides · Week 1 slides')."""
    return title if tag.replace("_", " ").lower() == title.lower() else f"{tag} · {title}"


def page():
    lectures = dict(c.rag().lectures)
    if st.session_state.get("summary_tag") not in lectures:
        st.session_state.summary_tag = next(iter(lectures))
    st.caption(t("sum_caption"))
    if len(lectures) > 1:
        # The choice lives in "summary_tag", so it survives switching tabs (a widget's own
        # state is dropped while the widget isn't on the page).
        st.session_state.summary_pick = st.session_state.summary_tag
        st.selectbox(t("sum_pick"), list(lectures), key="summary_pick",
                     format_func=lambda tag: lecture_name(tag, lectures[tag]),
                     on_change=lambda: st.session_state.update(
                         summary_tag=st.session_state.summary_pick))
    tag = st.session_state.summary_tag
    if len(lectures) == 1:
        st.subheader(lecture_name(tag, lectures[tag]), anchor=False)

    retry = st.session_state.pop("summary_retry", None) == tag
    entry = None if retry else find(tag)
    autogen = st.session_state.pop("summary_autogen", False)  # just uploaded, or asked for one
    if entry is None and (retry or autogen or st.button(
            t("sum_make", tag=tag), type="primary", icon=":material/summarize:")):
        if write(tag):
            st.rerun()  # redraw with the buttons under it
        return
    if entry:
        show(tag, lectures[tag], entry)
