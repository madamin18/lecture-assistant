"""Per-visitor state and helpers used by several tabs. State comes from st.session_state (one
per browser tab), never from module variables, which every visitor shares."""
import re

import pandas as pd
import streamlit as st
from openai import APIStatusError

import config
import rag as rag_module
from ui import resources

ASK = ":material/forum: Ask"
PRACTICE = ":material/quiz: Practice exam"
BROWSE = ":material/auto_stories: Browse lectures"
COMPARE = ":material/compare_arrows: Compare models"
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
    return "Auto" if ref in (None, AUTO) else rag_module.model_label(ref)


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
        return f"{title} · PDF page {page}"
    return f"{title} · slide {label}" + (f" (PDF page {page})" if label != page else "")


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
    st.toast("Thanks for the feedback!", icon=":material/favorite:")


def ask_lecture(tag: str):
    st.session_state.update(mode=ASK, lecture_filter=[tag])
    st.toast(f"Now searching only {tag}. Clear the filter in the sidebar to search everything.",
             icon=":material/filter_alt:")


def quiz_lecture(tag: str):
    st.session_state.update(mode=PRACTICE, quiz_lectures=[tag], quiz_autogen=True)


def api_error(e: APIStatusError):
    st.error(f"The AI service rejected the request (error {e.status_code}). "
             "If you run this site, check the API keys in `.env` or the app's Secrets.",
             icon=":material/key_off:")


# ---------- answers: how they were found, and their sources ----------
def chat_notes(messages: list) -> str:
    """The chat as Markdown study notes: each question, its answer, and the slides it cites."""
    lines = ["# Lecture Notes Assistant: chat", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"## {m['content']}", ""]
        else:
            lines += [m["content"], ""]
            cited = rag_module.cited_hits(m["content"], m.get("hits") or [])
            if cited:
                lines += ["Slides: " + "; ".join(f"{h['cite']} ({heading(h)})" for h in cited), ""]
    return "\n".join(lines)


def trace_label(t: dict) -> str:
    if t.get("lectures"):  # a lecture named in the question, or the sidebar filter
        n = sum(c["lecture"] in t["lectures"] for c in rag().chunks)
        scope = f"{n} slides of {', '.join(t['lectures'])}"
    else:
        scope = f"{t['n_slides']} slides"
    if t.get("cached"):
        return f"Searched {scope} · saved answer, shown instantly"
    label = f"Searched {scope} in {t['search_s']:.1f} s"
    if t.get("model"):
        label += f" · answered in {t['answer_s']:.1f} s"
    return label


def trace_steps(t: dict):
    """The steps that produced an answer, as a small timeline."""
    if t.get("translation"):
        with st.status("Translated the question for searching", type="step", state="complete"):
            st.caption(t["translation"])
    scope = (", ".join(t["lectures"]) + (" (named in the question)" if t.get("named") else
                                         " (sidebar filter)")) if t.get("lectures") else \
        f"all {t['n_lectures']} lectures"
    with st.status(f"Hybrid search: meaning + keywords, top {t['k']} slides",
                   type="step", state="complete"):
        st.caption(f"Searched {scope}.")
    for ref, reason in t.get("failed") or []:
        with st.status(f"{model_name(ref)} didn't answer", type="step", state="error"):
            st.caption(f"{reason[:1].upper()}{reason[1:]}, so the next model was asked.")
    if t.get("model"):
        with st.status(f"Answer written by {model_name(t['model'])}", type="step",
                       state="complete"):
            st.caption(f"{t['answer_s']:.1f} s, with a citation after each sentence.")
    elif t.get("refused_early"):
        with st.status("No slide was close enough, so the AI wasn't asked", type="step",
                       state="complete"):
            st.caption("The answer is the standard refusal.")
    if t.get("cached"):
        with st.status("Shown from saved answers", type="step", state="complete"):
            st.caption("This question was answered before with the same slides and settings, "
                       "so the saved answer is shown instantly instead of asking the AI again.")


def slides_table(hits: list):
    table = pd.DataFrame([{"Slide": h["cite"], "Title": heading(h), "Relevance": h["score"],
                           "Lecture": h["title"]} for h in hits])
    st.dataframe(table, hide_index=True, column_config={
        "Relevance": st.column_config.ProgressColumn(
            min_value=0.6, max_value=1.0, format="%.2f",
            help="Meaning-based similarity between the question and the slide")})


def show_sources(answer: str, hits: list):
    cited = rag_module.cited_hits(answer, hits)
    if cited:
        st.markdown("**:material/location_on: Found on** "
                    + " ".join(f":violet-badge[{h['cite']}]" for h in cited))
        with st.grid(2 if len(cited) > 1 else 1, gap="small"):
            for h in cited:
                show_slide(h)
    others = [h for h in hits if h not in cited]
    if others:
        label = "Other slides searched" if cited else "Slides searched"
        with st.expander(f"{label} ({len(others)})", icon=":material/search:"):
            slides_table(others)
