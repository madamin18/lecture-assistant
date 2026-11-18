"""Step 3: the website.  Run:  python -m streamlit run app.py

Three tabs:
  Ask              chat with the slides; each answer shows how it was found and the slides it
                   came from, with follow-up suggestions
  Practice exam    generated exam-style questions with instant feedback and a running score
  Browse lectures  lecture cards and a slide viewer, with shortcuts to ask about a slide or
                   get quizzed on a lecture

Visitors can upload their own lecture PDFs (kept in memory for their browser tab only, never
saved on the server). With no uploads, the site uses the course slides in index/ (built by
ingest.py).
"""
import hashlib
import importlib
import re
import time
from pathlib import Path

import pandas as pd
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

st.set_page_config(page_title="Lecture Notes Assistant", page_icon=":material/school:")

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


@st.cache_data(max_entries=400, show_spinner=False)
def slide_image(pdf_ref: str, page: int, doc_key: str, _data: bytes | None) -> bytes:
    # doc_key identifies the PDF's content; _data (leading underscore = not hashed) is the upload.
    return render_page(pdf_ref, page, _data)


@st.cache_data(show_spinner=False)
def pdf_page_count(pdf_ref: str, doc_key: str, _data: bytes | None) -> int:
    return page_count(pdf_ref, _data)


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
             "If you run this site, check the API key in `.env` or the app's Secrets.",
             icon=":material/key_off:")


# ---------- constants ----------
ASK = ":material/forum: Ask"
PRACTICE = ":material/quiz: Practice exam"
BROWSE = ":material/auto_stories: Browse lectures"

DEMO_QUESTIONS = {  # one per part of the course; labels are what the chips show
    ":material/help: What is a hallucination?": "What is a hallucination?",
    ":material/thermostat: Does temperature make answers more accurate?":
        "Does raising the temperature make the model's answers more accurate?",
    ":material/analytics: How is recall@k calculated?": "How is recall@k calculated?",
    ":material/account_tree: Workflow vs agent":
        "What is the difference between a fixed workflow and an agent?",
    ":material/translate: 환각이란 무엇인가요?": "환각이란 무엇인가요?",
}
KINDS = {"mixed": "Mixed", "mcq": "Multiple choice", "short": "Short answer"}
LANGS = {"same": "Same as slides", "en": "English", "ko": "한국어"}
LETTERS = "ABCDE"


# ---------- sidebar, part 1: uploads ----------
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
        rag, notes = load_demo(VERSION), []
    st.session_state.update(corpus_key=key, version=VERSION, rag=rag, notes=notes,
                            messages=[], quiz=None, lecture_filter=[], quiz_lectures=[])
for name, default in (("messages", []), ("quiz", None), ("lecture_filter", []),
                      ("quiz_lectures", []), ("mode", ASK)):
    st.session_state.setdefault(name, default)
rag = st.session_state.rag

# Jumps requested from the slide viewer dialog (applied before any widget is drawn).
goto = st.session_state.pop("goto", None)
if goto:
    where, payload = goto
    if where == "ask":
        st.session_state.update(mode=ASK, pending_question=payload)
    elif where == "quiz":
        st.session_state.update(mode=PRACTICE, quiz_lectures=[payload], quiz_autogen=True)


# ---------- helpers ----------
def slide_title(text: str, words: int = 9) -> str:
    """First words of a slide (its title, usually), without the footer or a leading all-caps
    section label such as "REVIEW AND REFERENCES" or "LANGGRAPH LABS"."""
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


# Slides that make poor follow-up questions.
SKIP_TITLES = re.compile(r"^(learning outcomes|references?|sources|exit questions|agenda|"
                         r"outline|roadmap|questions)", re.IGNORECASE)


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


def page_index(pdf: str) -> dict:
    """PDF page -> (printed label, chunk) for one lecture file."""
    out = {}
    for c in rag.chunks:
        if c["pdf"] == pdf:
            for page, label in zip(c["pages"], c.get("labels") or [None] * len(c["pages"])):
                out[page] = (label, c)
    return out


def lecture_file(tag: str) -> str:
    return next(c["pdf"] for c in rag.chunks if c["lecture"] == tag)


def thanks():
    st.toast("Thanks for the feedback!", icon=":material/favorite:")


def ask_lecture(tag: str):
    st.session_state.update(mode=ASK, lecture_filter=[tag])
    st.toast(f"Now searching only {tag}. Clear the filter in the sidebar to search everything.",
             icon=":material/filter_alt:")


def quiz_lecture(tag: str):
    st.session_state.update(mode=PRACTICE, quiz_lectures=[tag], quiz_autogen=True)


# ---------- sidebar, part 2: search settings ----------
with st.sidebar:
    if rag is not None and len(rag.lectures) > 1:
        st.pills("Search only these lectures", [t for t, _ in rag.lectures],
                 selection_mode="multi", key="lecture_filter",
                 help="Leave empty to search every lecture. Naming a lecture in your question "
                      "(\"in Lesson 3\", \"3강\") also works.")
    k = st.slider("Slides read per answer", 3, 12, config.TOP_K,
                  help="How many of the best-matching slides the assistant reads before answering.")
    if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.messages = []
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
source = "uploaded" if files else "course"
st.markdown(
    f":violet-badge[:material/menu_book: {n_lectures} {source} lecture{'s' * (n_lectures > 1)}] "
    f":blue-badge[:material/slideshow: {len(rag.chunks)} slides] "
    ":green-badge[:material/translate: English · 한국어] "
    ":orange-badge[:material/verified: Answers cite their slides]")
if not files:
    st.caption("Demo: the lectures of *Design of Generative AI Application Systems*. "
               "Upload your own PDFs in the sidebar to use those instead.")

mode = st.segmented_control("Mode", [ASK, PRACTICE, BROWSE], required=True, key="mode",
                            label_visibility="collapsed", width="stretch")


# ======================= Ask =======================
def trace_label(t: dict) -> str:
    label = f"Searched {t['n_slides']} slides in {t['search_s']:.1f} s"
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
    if t.get("model"):
        with st.status(f"Answer written by {t['model']}", type="step", state="complete"):
            st.caption(f"{t['answer_s']:.1f} s, with a citation after each sentence.")
    elif t.get("refused_early"):
        with st.status("No slide was close enough, so the AI wasn't asked", type="step",
                       state="complete"):
            st.caption("The answer is the standard refusal.")


def show_sources(answer: str, hits: list):
    cited = cited_hits(answer, hits)
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
            table = pd.DataFrame([{"Slide": h["cite"], "Lecture": h["title"],
                                   "Relevance": h["score"], "Starts with": slide_title(h["text"], 12)}
                                  for h in others])
            st.dataframe(table, hide_index=True, column_config={
                "Relevance": st.column_config.ProgressColumn(
                    min_value=0.6, max_value=1.0, format="%.2f",
                    help="Meaning-based similarity between the question and the slide")})


def next_questions(answer: str, hits: list) -> dict:
    """Up to 3 follow-ups built from the other slides the search found (no extra AI call)."""
    cited = cited_hits(answer, hits)
    pool = [h for h in hits if h not in cited] or hits
    out = {}
    for h in pool:
        title = slide_title(h["text"], 8)
        if (1 in h["pages"] or SKIP_TITLES.match(title) or len(title) < 15
                or any(title.rstrip("…") in v for v in out.values())):
            continue
        out[f":material/arrow_forward: {h['lecture']} · {title}"] = \
            f'Explain "{title.rstrip("…")}" from {h["lecture"]}.'
        if len(out) == 3:
            break
    return out


def pick_chip(widget_key: str, options: dict):
    """Turn a clicked chip into the next question, then unselect it."""
    st.session_state.pending_question = options[st.session_state[widget_key]]
    st.session_state[widget_key] = None


def starter_chips() -> dict:
    if not files:
        return DEMO_QUESTIONS
    return {f":material/summarize: Key takeaways of {tag}": f"What are the key takeaways of {tag}?"
            for tag, _ in rag.lectures[:4]}


def ask_page():
    messages = st.session_state.messages
    for i, m in enumerate(messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] != "assistant":
                continue
            if m.get("trace"):
                with st.expander(trace_label(m["trace"]), type="compact",
                                 icon=":material/manage_search:"):
                    trace_steps(m["trace"])
            if m.get("hits"):
                show_sources(m["content"], m["hits"])
            if i == len(messages) - 1 and m.get("hits"):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.feedback("thumbs", key=f"fb_{len(messages)}", on_change=thanks)
                options = next_questions(m["content"], m["hits"])
                if options:
                    st.pills("Ask next", list(options), key=f"next_{len(messages)}",
                             on_change=pick_chip, args=(f"next_{len(messages)}", options))

    question = (st.chat_input("Ask a question about the slides (English or 한국어)",
                              submit_mode="disable")
                or st.session_state.pop("pending_question", None))

    if not messages:
        welcome = st.empty()
        with welcome.container(border=True):
            st.subheader(":material/waving_hand: What would you like to know?", anchor=False)
            st.markdown("Ask anything about the lectures, in English or Korean. Every answer "
                        "shows the slides it came from.")
            st.caption("Tip: name a lecture (\"in Lesson 3\", \"3강\") or use the sidebar "
                       "filter to search only that one.")
            options = starter_chips()
            st.pills("Try asking", list(options), key="starter",
                     on_change=pick_chip, args=("starter", options))
        if question:
            welcome.empty()  # hide the welcome card before the first answer renders
    elif st.session_state.lecture_filter:
        st.caption(f":material/filter_alt: Searching only {', '.join(st.session_state.lecture_filter)}"
                   " (change this in the sidebar).")

    if not question:
        return
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        trace = {"n_slides": len(rag.chunks), "n_lectures": n_lectures, "k": k}
        try:
            t0 = time.time()
            with st.status(":shimmer[Searching the slides…]", type="compact") as status:
                prepared, hits = rag.prepare(question, k, history,
                                             st.session_state.lecture_filter or None)
                trace.update(search_s=time.time() - t0, **rag.last_search)
                trace_steps({**trace, "model": None})
                status.update(label=trace_label(trace), state="complete")
            if prepared is None:  # nothing close enough in the slides: refuse without the LLM
                answer = refusal_for(question)
                trace["refused_early"] = True
                st.markdown(answer)
            else:
                t1 = time.time()
                answer = st.write_stream(rag.stream(prepared))  # text appears as it's written
                trace.update(answer_s=time.time() - t1, model=rag.last_model)
        except APIStatusError as e:
            api_error(e)
            return
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "hits": hits, "trace": trace}]
    st.rerun()  # redraw with slides, feedback and follow-up suggestions


# ======================= Practice exam =======================
def make_quiz(n: int, kind: str, lectures: list, topic: str, lang: str = "same"):
    with st.spinner("Writing practice questions from your slides…"):
        try:
            questions = rag.generate_quiz(n, kind, lectures, topic, lang)
        except APIStatusError as e:
            api_error(e)
            return
    if questions is None:
        st.warning(config.BUSY_MESSAGE, icon=":material/hourglass_top:")
        return
    if not questions:
        st.warning("Couldn't write questions from these slides this time. Please try again"
                   + (" or try a different topic." if topic else "."))
        return
    st.session_state.quiz_count = st.session_state.get("quiz_count", 0) + 1
    st.session_state.quiz = {
        "id": st.session_state.quiz_count, "questions": questions,
        "responses": {}, "revealed": set(), "self": {}, "celebrated": False,
        "settings": {"n": n, "kind": kind, "lectures": lectures, "topic": topic, "lang": lang},
    }
    st.toast(f"{len(questions)} questions ready. Good luck!", icon=":material/quiz:")


def answer_key(qid: int, i: int) -> str:
    return f"quiz{qid}_q{i}"


def record_choice(i: int):
    quiz = st.session_state.quiz
    quiz["responses"][i] = st.session_state[answer_key(quiz["id"], i)]


def reveal(i: int):
    st.session_state.quiz["revealed"].add(i)


def self_grade(i: int):
    quiz = st.session_state.quiz
    quiz["self"][i] = st.session_state[f"self_{quiz['id']}_{i}"]


def retry_quiz():
    """Same questions, fresh answers (new widget keys)."""
    quiz = st.session_state.quiz
    st.session_state.quiz_count = st.session_state.get("quiz_count", 0) + 1
    quiz.update(id=st.session_state.quiz_count, responses={}, revealed=set(), self={},
                celebrated=False)


def quiz_score(quiz) -> tuple[int, int]:
    """(answered, correct). Short answers count once the student grades themselves."""
    answered = correct = 0
    for i, q in enumerate(quiz["questions"]):
        if q["type"] == "mcq" and i in quiz["responses"]:
            answered += 1
            correct += quiz["responses"][i] == q["options"][q["answer"]]
        elif q["type"] == "short" and quiz["self"].get(i) is not None:
            answered += 1
            correct += quiz["self"][i] == 1
    return answered, correct


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


def show_source(q: dict):
    with st.expander(f"Source: {q['hit']['cite']}", icon=":material/location_on:"):
        show_slide(q["hit"])


def show_question(quiz: dict, i: int, q: dict):
    qid = quiz["id"]
    kind = "Multiple choice" if q["type"] == "mcq" else "Short answer"
    st.markdown(f":violet-badge[Question {i + 1}] :gray-badge[{kind}] "
                f":blue-badge[{q['hit']['lecture']}]")
    st.markdown(f"**{q['question']}**")
    if q["type"] == "mcq":
        chosen = quiz["responses"].get(i)
        opts = q["options"]
        st.radio("Your answer", opts, key=answer_key(qid, i),
                 index=opts.index(chosen) if chosen in opts else None,
                 format_func=lambda o, opts=opts: f"{LETTERS[opts.index(o)]}) {o}",
                 disabled=chosen is not None, on_change=record_choice, args=(i,),
                 label_visibility="collapsed")
        if chosen is not None:
            right = opts[q["answer"]]
            if chosen == right:
                st.success(f"Correct: **{LETTERS[q['answer']]}) {right}**",
                           icon=":material/check_circle:")
            else:
                st.error(f"You picked **{LETTERS[opts.index(chosen)]}**. The answer is "
                         f"**{LETTERS[q['answer']]}) {right}**", icon=":material/cancel:")
            if q["explanation"]:
                st.caption(q["explanation"])
            show_source(q)
    else:
        revealed = i in quiz["revealed"]
        st.text_area("Your answer", key=answer_key(qid, i), disabled=revealed,
                     placeholder="Write your answer, then reveal the model answer to compare.",
                     label_visibility="collapsed")
        if not revealed:
            st.button("Reveal model answer", key=f"reveal_{qid}_{i}", on_click=reveal,
                      args=(i,), icon=":material/visibility:")
        else:
            st.info(f"**Model answer:** {q['answer']}", icon=":material/lightbulb:")
            if q["explanation"]:
                st.caption(q["explanation"])
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("Did you get it right?")
                st.feedback("thumbs", key=f"self_{qid}_{i}", on_change=self_grade, args=(i,))
            show_source(q)


def show_quiz(quiz: dict):
    qs = quiz["questions"]
    answered, correct = quiz_score(quiz)
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.metric("Answered", f"{answered} / {len(qs)}")
        b.metric("Correct", correct)
        c.metric("Score", f"{round(100 * correct / answered)}%" if answered else "–")
        st.progress(answered / len(qs))

    for i, q in enumerate(qs):
        with st.container(border=True):
            show_question(quiz, i, q)

    if answered == len(qs):
        pct = round(100 * correct / len(qs))
        if pct >= 80:
            st.success(f"Finished: {correct} / {len(qs)} ({pct}%). Excellent work!",
                       icon=":material/emoji_events:")
            if not quiz["celebrated"]:
                quiz["celebrated"] = True
                st.balloons()
        else:
            st.info(f"Finished: {correct} / {len(qs)} ({pct}%). Review the source slides for "
                    "the ones you missed, then try again.", icon=":material/school:")

    with st.container(horizontal=True):
        st.button("New questions", icon=":material/refresh:", type="primary",
                  on_click=lambda: st.session_state.update(quiz_regen=True))
        st.button("Try again", icon=":material/replay:", on_click=retry_quiz)
        st.download_button("Download study sheet", study_sheet(quiz),
                           file_name="practice_questions.md", mime="text/markdown",
                           icon=":material/download:", on_click="ignore")


def practice_page():
    lectures = dict(rag.lectures)
    with st.form("quiz_settings"):
        st.subheader(":material/edit_note: Build a practice exam", anchor=False)
        chosen = []
        if len(lectures) > 1:
            chosen = st.pills("Lectures (leave empty for all)", list(lectures),
                              selection_mode="multi", key="quiz_lectures",
                              format_func=lambda t: f"{t} {lectures[t]}")
        topic = st.text_input("Topic (optional)",
                              placeholder="e.g. hallucinations. Leave empty to cover everything.")
        with st.container(horizontal=True):
            n = st.segmented_control("Questions", [5, 10, 15], default=5, required=True)
            kind = st.segmented_control("Type", list(KINDS), format_func=KINDS.get,
                                        default="mixed", required=True)
        lang = st.segmented_control("Language", list(LANGS), format_func=LANGS.get,
                                    default="same", required=True)
        go = st.form_submit_button("Generate questions", type="primary", width="stretch",
                                   icon=":material/auto_awesome:")

    if go:
        make_quiz(n, kind, chosen, topic, lang)
    elif st.session_state.pop("quiz_autogen", False):
        make_quiz(5, "mixed", st.session_state.quiz_lectures, "", "same")
    elif st.session_state.pop("quiz_regen", False) and st.session_state.quiz:
        make_quiz(**st.session_state.quiz["settings"])

    if st.session_state.quiz:
        show_quiz(st.session_state.quiz)
    else:
        st.caption(":material/info: Questions are checked against the slides they cite. "
                   "Answer one and you'll see right away whether you got it.")


# ======================= Browse lectures =======================
@st.dialog("Slide viewer", width="large", icon=":material/slideshow:")
def slide_viewer(tag: str):
    title = dict(rag.lectures)[tag]
    pdf = lecture_file(tag)
    data = rag.pdf_data.get(pdf)
    pages = page_index(pdf)
    total = pdf_page_count(pdf, key, data)
    st.markdown(f":violet-badge[{tag}] **{title}** · {total} slides")
    page = st.select_slider(
        "Slide", options=list(range(1, total + 1)), key=f"viewer_{tag}",
        format_func=lambda p: f"slide {pages.get(p, (None,))[0] or p}")
    label, chunk = pages.get(page, (None, None))
    with st.container(horizontal=True):  # above the slide, so no scrolling is needed
        if chunk is not None and st.button("Ask about this slide", type="primary",
                                           icon=":material/forum:"):
            st.session_state.goto = ("ask", f'Explain this slide from {tag} (slide {label or page}): '
                                            f'"{slide_title(chunk["text"], 12).rstrip("…")}"')
            st.rerun()
        if st.button("Quiz me on this lecture", icon=":material/quiz:"):
            st.session_state.goto = ("quiz", tag)
            st.rerun()
    st.image(slide_image(pdf, page, key, data), caption=slide_caption(title, page, label))


def browse_page():
    st.caption("Open a lecture to flip through its slides, ask about any slide, or get quizzed.")
    with st.grid("auto", min_column_width=210, gap="small"):
        for tag, title in rag.lectures:
            pdf = lecture_file(tag)
            data = rag.pdf_data.get(pdf)
            with st.container(border=True, height="stretch"):
                st.image(slide_image(pdf, 1, key, data))
                st.markdown(f":violet-badge[{tag}] **{title}**")
                st.caption(f"{pdf_page_count(pdf, key, data)} slides")
                st.space("stretch")
                if st.button("View slides", key=f"view_{tag}", icon=":material/slideshow:",
                             width="stretch"):
                    slide_viewer(tag)
                with st.container(horizontal=True, gap="small"):
                    st.button("Ask", key=f"ask_{tag}", icon=":material/forum:",
                              on_click=ask_lecture, args=(tag,), width="stretch")
                    st.button("Quiz", key=f"quiz_{tag}", icon=":material/quiz:",
                              on_click=quiz_lecture, args=(tag,), width="stretch")


if mode == PRACTICE:
    practice_page()
elif mode == BROWSE:
    browse_page()
else:
    ask_page()
