"""Step 3: the website.  Run:  streamlit run app.py

Two modes:
  Ask            chat with the slides; every answer shows the slides it came from
  Practice exam  generated exam-style questions (multiple choice / short answer) with
                 answers, explanations and the source slide

Visitors upload their own lecture PDFs (kept in memory for their browser tab only,
never saved on the server). With no uploads, the site uses the demo slides in index/
(built by ingest.py).
"""
import hashlib
import importlib
import time
from pathlib import Path

import streamlit as st
from openai import APIStatusError
from sentence_transformers import SentenceTransformer

import config
import ingest
import rag as rag_module


def code_version() -> str:
    """Fingerprint of the search/answer code and settings."""
    h = hashlib.sha1()
    for name in ("rag.py", "config.py", "ingest.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()[:12]


def load_code() -> str:
    """A deploy that only changes rag.py / config.py / ingest.py doesn't restart the server, and
    Python keeps running the old modules (rag.py keeps reading the old config). Whenever the
    files differ from what is loaded, reload them in dependency order. Returns an id that
    changes on every reload; cached objects below take it as an argument, so new code always
    gets new objects (also when a change is reverted)."""
    version = code_version()
    if getattr(rag_module, "LOADED_VERSION", None) != version:
        for module in (config, ingest, rag_module):
            importlib.reload(module)
        rag_module.LOADED_VERSION = version
        rag_module.LOAD_ID = f"{version}-{time.time_ns()}"
    return rag_module.LOAD_ID


VERSION = load_code()

from ingest import build_index, page_count  # noqa: E402  (imported after the reload on purpose)
from rag import SlideRAG, cited_hits, make_client, refusal_for, render_page  # noqa: E402

st.set_page_config(page_title="Lecture Notes Assistant", page_icon="📚")

if not config.LLM_API_KEY:
    st.error("No API key found. Add GEMINI_API_KEY to `.env` (local) or to the app's "
             "Secrets on Streamlit Cloud (Manage app -> Settings -> Secrets).")
    st.stop()


# ---------- shared by all visitors (loaded once per server) ----------
# Cached objects take VERSION (from load_code) as an argument, so new code gets new objects.
@st.cache_resource(show_spinner="Loading the search model…")
def load_embedder():
    return SentenceTransformer(config.EMBED_MODEL)


@st.cache_resource
def load_client(version: str):
    return make_client()


@st.cache_resource(show_spinner=False)
def load_demo(version: str):
    try:
        return SlideRAG(embedder=load_embedder(), client=load_client(version))
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
                   embedder=load_embedder(), client=load_client(VERSION))
    return rag, notes


def corpus_key(files) -> str:
    if not files:
        return "demo"
    return "-".join(hashlib.sha1(f.getvalue()).hexdigest()[:12] for f in files)


def api_error(e: APIStatusError):
    st.error(f"The AI service rejected the request (error {e.status_code}). "
             "If you run this site, check the API key in `.env` or the app's Secrets.")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Your lecture slides")
    files = st.file_uploader(
        "Upload lecture PDFs", type="pdf", accept_multiple_files=True,
        help="Only you can search the files you upload. They stay in memory for this "
             "browser tab and are never saved on the server.")
    st.markdown("---")
    k = st.slider("Slides to search", 3, 10, config.TOP_K,
                  help="How many slides the assistant reads before answering a question.")
    show_images = st.toggle("Show previews of other searched slides", value=False)
    if st.button("Clear chat", icon=":material/delete_sweep:"):
        st.session_state.messages = []
    st.markdown("---")
    st.caption("Answers and practice questions come only from the slides and can still be "
               "wrong. Check the slide shown with each one before relying on it.")

# Re-index only when the set of uploaded files (or the app's code) changes;
# a new set starts a new chat and quiz.
key = corpus_key(files)
if st.session_state.get("corpus_key") != key or st.session_state.get("version") != VERSION:
    if files:
        rag, notes = index_uploads(files)
    else:
        rag, notes = load_demo(VERSION), []
    st.session_state.update(corpus_key=key, version=VERSION, rag=rag, notes=notes,
                            messages=[], quiz=None)
st.session_state.setdefault("quiz", None)  # sessions started before the quiz existed
rag = st.session_state.rag

# ---------- main page ----------
st.title("Lecture Notes Assistant")
st.caption("Ask questions about your lecture slides or practise for the exam. "
           "Every answer shows the slide it came from.")

for level, text in st.session_state.notes:
    getattr(st, level)(text)

if rag is None:
    if not files:
        st.info("Upload one or more lecture PDFs in the sidebar to get started.")
    st.stop()

if not files:
    lectures = rag.lectures
    if len(lectures) == 1:
        st.info(f"**Demo mode:** using the sample lecture *{lectures[0][1]}*. "
                "Upload your own PDFs in the sidebar to use those instead.")
    else:
        names = " · ".join(f"{lec} {title}" for lec, title in lectures)
        st.info(f"**Demo mode:** all {len(lectures)} lectures of the course "
                f"({len(rag.chunks)} slides): {names}.  \n"
                "Tip: name a lecture (\"in Lesson 3\", \"3강\") to search only that one. "
                "Upload your own PDFs in the sidebar to use those instead.")
else:
    n_files = len({c["pdf"] for c in rag.chunks})
    st.success(f"Using {n_files} uploaded file(s), {len(rag.chunks)} slides.  \n"
               "Tip: name a lecture (\"in Lesson 3\", \"L03\") to search only that one.")


def slide_caption(title: str, page: int, label) -> str:
    """'Intro · slide 59 (PDF page 56)': the number printed on the slide, plus where it is in
    the PDF when that differs (exported decks often skip hidden slides)."""
    if label is None:
        return f"{title} · PDF page {page}"
    return f"{title} · slide {label}" + (f" (PDF page {page})" if label != page else "")


def show_slide(h):
    data = rag.pdf_data.get(h["pdf"])
    labels = h.get("labels") or [None] * len(h["pages"])
    for page, label in list(zip(h["pages"], labels))[-2:]:  # merged chunks: content is last
        st.image(slide_image(h["pdf"], page, key, data),
                 caption=slide_caption(h["title"], page, label))


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


ASK, PRACTICE = ":material/chat: Ask a question", ":material/quiz: Practice exam"
mode = st.segmented_control("Mode", [ASK, PRACTICE], default=ASK, required=True,
                            key="mode", label_visibility="collapsed")


# ======================= Ask =======================
# One-click starter questions for the demo slides (hidden once the chat starts).
DEMO_QUESTIONS = {  # one per part of the course; labels are what the chips show
    ":material/help: What is a hallucination?": "What is a hallucination?",
    ":material/thermostat: Does temperature make answers more accurate?":
        "Does raising the temperature make the model's answers more accurate?",
    ":material/analytics: How is recall@k calculated?": "How is recall@k calculated?",
    ":material/account_tree: Workflow vs agent":
        "What is the difference between a fixed workflow and an agent?",
    ":material/translate: 환각이란 무엇인가요?": "환각이란 무엇인가요?",
}


def use_suggestion():
    # Turn the clicked chip into a question, then unselect it so "Clear chat" shows it again.
    st.session_state.pending_question = DEMO_QUESTIONS[st.session_state.demo_pick]
    st.session_state.demo_pick = None


def ask_page():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("hits"):
                show_sources(m["content"], m["hits"])
            if m.get("model"):
                st.caption(f"Answered by {m['model']}")

    question = (st.chat_input("Ask a question about the slides (English or 한국어)",
                              submit_mode="disable")
                or st.session_state.pop("pending_question", None))

    if not files and not st.session_state.messages:
        suggestions = st.empty()
        suggestions.pills("Try asking:", list(DEMO_QUESTIONS), key="demo_pick",
                          on_change=use_suggestion)
        if question:
            suggestions.empty()  # hide the chips before the first answer renders

    if not question:
        return
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the slides…"):
                messages, hits = rag.prepare(question, k, history)
            if messages is None:  # nothing close enough in the slides: refuse without the LLM
                answer = refusal_for(question)
                st.markdown(answer)
            else:
                answer = st.write_stream(rag.stream(messages))  # text appears as it's written
        except APIStatusError as e:
            api_error(e)
            return
        show_sources(answer, hits)
        if rag.last_model:
            st.caption(f"Answered by {rag.last_model}")
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "hits": hits, "model": rag.last_model}]


# ======================= Practice exam =======================
KINDS = {"mixed": "Mixed", "mcq": "Multiple choice", "short": "Short answer"}
LANGS = {"same": "Same as slides", "en": "English", "ko": "한국어"}
LETTERS = "ABCDE"


def make_quiz(n: int, kind: str, lectures: list, topic: str, lang: str = "same"):
    with st.spinner("Writing practice questions from your slides…"):
        try:
            questions = rag.generate_quiz(n, kind, lectures, topic, lang)
        except APIStatusError as e:
            api_error(e)
            return
    if questions is None:
        st.warning(config.BUSY_MESSAGE)
        return
    if not questions:
        st.warning("Couldn't write questions from these slides this time. Please try again"
                   + (" or try a different topic." if topic else "."))
        return
    st.session_state.quiz_count = st.session_state.get("quiz_count", 0) + 1
    st.session_state.quiz = {
        "id": st.session_state.quiz_count, "questions": questions, "checked": False,
        "responses": {},
        "settings": {"n": n, "kind": kind, "lectures": lectures, "topic": topic, "lang": lang},
    }


def answer_key(qid: int, i: int) -> str:
    return f"quiz{qid}_q{i}"


def check_answers():
    """Form callback: runs before the page redraws, so feedback appears straight away."""
    quiz = st.session_state.quiz
    quiz["checked"] = True
    quiz["responses"] = {i: st.session_state.get(answer_key(quiz["id"], i))
                         for i in range(len(quiz["questions"]))}


def study_sheet(quiz) -> str:
    """Printable Markdown: questions first, answer key at the end."""
    lines = ["# Practice questions", ""]
    for i, q in enumerate(quiz["questions"], 1):
        lines.append(f"{i}. {q['question']}")
        if q["type"] == "mcq":
            lines += [f"   {LETTERS[j]}) {opt}" for j, opt in enumerate(q["options"])]
        lines.append("")
    lines += ["---", "", "## Answers", ""]
    for i, q in enumerate(quiz["questions"], 1):
        if q["type"] == "mcq":
            ans = f"{LETTERS[q['answer']]}) {q['options'][q['answer']]}"
        else:
            ans = q["answer"]
        lines.append(f"{i}. **{ans}** {q['explanation']} [{q['hit']['cite']}]")
    return "\n".join(lines) + "\n"


def show_feedback(q: dict, response):
    if q["type"] == "mcq":
        correct = q["options"][q["answer"]]
        if response == correct:
            st.success("Correct!", icon=":material/check_circle:")
        elif response is None:
            st.warning(f"Not answered. The answer is **{LETTERS[q['answer']]}) {correct}**")
        else:
            st.error(f"Not quite. The answer is **{LETTERS[q['answer']]}) {correct}**",
                     icon=":material/cancel:")
    else:
        st.info(f"**Model answer:** {q['answer']}", icon=":material/lightbulb:")
    if q["explanation"]:
        st.caption(q["explanation"])
    with st.expander(f"📍 Source: {q['hit']['cite']}"):
        show_slide(q["hit"])


def show_quiz(quiz):
    qs, qid = quiz["questions"], quiz["id"]
    if quiz["checked"]:
        mcqs = [(i, q) for i, q in enumerate(qs) if q["type"] == "mcq"]
        right = sum(quiz["responses"].get(i) == q["options"][q["answer"]] for i, q in mcqs)
        parts = []
        if mcqs:
            parts.append(f"**{right} / {len(mcqs)}** multiple-choice questions right.")
        if len(mcqs) < len(qs):
            parts.append("Compare your short answers with the model answers below.")
        st.markdown(" ".join(parts))

    with st.form(f"quiz_form_{qid}", border=False):
        for i, q in enumerate(qs):
            saved = quiz["responses"].get(i)
            with st.container(border=True):
                st.markdown(f"**{i + 1}. {q['question']}**")
                if q["type"] == "mcq":
                    st.radio("Your answer", q["options"], key=answer_key(qid, i),
                             index=q["options"].index(saved) if saved in q["options"] else None,
                             format_func=lambda o, opts=q["options"]: f"{LETTERS[opts.index(o)]}) {o}",
                             label_visibility="collapsed")
                else:
                    st.text_area("Your answer", value=saved or "", key=answer_key(qid, i),
                                 placeholder="Write your answer, then check it against the model answer.",
                                 label_visibility="collapsed")
                if quiz["checked"]:
                    show_feedback(q, quiz["responses"].get(i))
        st.form_submit_button("Check answers", type="primary", icon=":material/fact_check:",
                              on_click=check_answers)

    with st.container(horizontal=True):
        st.button("New questions", icon=":material/refresh:",
                  on_click=lambda: st.session_state.update(quiz_regen=True))
        st.download_button("Download as study sheet", study_sheet(quiz),
                           file_name="practice_questions.md", mime="text/markdown",
                           icon=":material/download:", on_click="ignore")


def practice_page():
    st.markdown("Generate exam-style questions from the slides, answer them, then check "
                "yourself. Each answer links to the slide it comes from.")
    lectures = dict(rag.lectures)
    with st.form("quiz_settings"):
        chosen = []
        if len(lectures) > 1:
            chosen = st.multiselect("Lectures", list(lectures), placeholder="All lectures",
                                    format_func=lambda l: f"{l} – {lectures[l]}")
        topic = st.text_input("Topic (optional)",
                              placeholder="e.g. hallucinations. Leave empty to cover everything.")
        with st.container(horizontal=True):
            n = st.segmented_control("Number of questions", [5, 10, 15], default=5, required=True)
            kind = st.segmented_control("Question type", list(KINDS), format_func=KINDS.get,
                                        default="mixed", required=True)
        lang = st.segmented_control("Language", list(LANGS), format_func=LANGS.get,
                                    default="same", required=True)
        go = st.form_submit_button("Generate questions", type="primary",
                                   icon=":material/auto_awesome:")

    if go:
        make_quiz(n, kind, chosen, topic, lang)
    elif st.session_state.pop("quiz_regen", False) and st.session_state.quiz:
        make_quiz(**st.session_state.quiz["settings"])

    if st.session_state.quiz:
        show_quiz(st.session_state.quiz)


if mode == PRACTICE:
    practice_page()
else:
    ask_page()
