"""Practice exam tab: exam-style questions from the slides, with instant feedback, a running
score, and help on the ones missed."""
import streamlit as st
from openai import APIStatusError

import config
import rag as rag_module
from ui import common as c

KINDS = {"mixed": "Mixed", "mcq": "Multiple choice", "short": "Short answer"}
LANGS = {"same": "Same as slides", "en": "English", "ko": "한국어"}
LETTERS = "ABCDE"


def make_quiz(n: int, kind: str, lectures: list, topic: str, lang: str = "same"):
    info = {}
    with st.spinner("Writing practice questions from your slides…"):
        try:
            questions = c.rag().generate_quiz(n, kind, lectures, topic, lang, info=info,
                                              model=c.chosen_model())
        except APIStatusError as e:
            c.api_error(e)
            return
    if questions is None:
        st.warning(config.BUSY_MESSAGE, icon=":material/hourglass_top:")
        return
    if not questions:
        st.warning("Couldn't write questions from these slides this time. Please try again"
                   + (" or try a different topic." if topic else ".")
                   + (" A different AI model may do better." if c.chosen_model() else ""))
        return
    st.session_state.quiz_count = st.session_state.get("quiz_count", 0) + 1
    st.session_state.quiz = {
        "id": st.session_state.quiz_count, "questions": questions, "model": info.get("model"),
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


def result(quiz, i: int, q: dict) -> bool | None:
    """True/False once answered, None before. Short answers count once self-graded."""
    if q["type"] == "mcq":
        chosen = quiz["responses"].get(i)
        return None if chosen is None else chosen == q["options"][q["answer"]]
    graded = quiz["self"].get(i)
    return None if graded is None else graded == 1


def quiz_score(quiz) -> tuple[int, int]:
    """(answered, correct)."""
    results = [result(quiz, i, q) for i, q in enumerate(quiz["questions"])]
    return sum(r is not None for r in results), sum(r is True for r in results)


def retry_quiz(missed_only: bool = False):
    """Same questions (or only the ones answered wrong), fresh answers (new widget keys)."""
    quiz = st.session_state.quiz
    if missed_only:
        quiz["questions"] = [q for i, q in enumerate(quiz["questions"])
                             if result(quiz, i, q) is False]
    st.session_state.quiz_count = st.session_state.get("quiz_count", 0) + 1
    quiz.update(id=st.session_state.quiz_count, responses={}, revealed=set(), self={},
                celebrated=False)


def explain_question(q: dict):
    """Ask the assistant to explain a practice question (opens the Ask tab)."""
    right = q["options"][q["answer"]] if q["type"] == "mcq" else q["answer"]
    lec = q["hit"]["lecture"]  # naming the lecture makes the search look only there
    if rag_module.language(q["question"]) == "ko":
        ask = f'{lec}: {q["question"]} 정답이 "{right}"인 이유를 설명해 주세요.'
    else:
        ask = f'{lec}: {q["question"]} Explain why the answer is "{right}".'
    st.session_state.update(mode=c.ASK, pending_question=ask)


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


def show_source(q: dict, qid: int, i: int, explain: bool = False):
    with st.expander(f"Source: {q['hit']['cite']} · {c.shorten(c.heading(q['hit']), 60)}",
                     icon=":material/location_on:"):
        c.show_slide(q["hit"])
    if explain:
        st.button("Explain this to me", key=f"explain_{qid}_{i}", icon=":material/forum:",
                  on_click=explain_question, args=(q,),
                  help="Ask the assistant about this question; it answers from the slides.")


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
            show_source(q, qid, i, explain=chosen != right)
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
            show_source(q, qid, i, explain=quiz["self"].get(i) == 0)


def show_quiz(quiz: dict):
    qs = quiz["questions"]
    answered, correct = quiz_score(quiz)
    with st.container(border=True):
        a, b, col = st.columns(3)
        a.metric("Answered", f"{answered} / {len(qs)}")
        b.metric("Correct", correct)
        col.metric("Score", f"{round(100 * correct / answered)}%" if answered else "–")
        st.progress(answered / len(qs))
        if quiz.get("model"):
            st.caption(f":material/smart_toy: Questions written by {c.model_name(quiz['model'])}")

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

    missed = sum(result(quiz, i, q) is False for i, q in enumerate(qs))
    with st.container(horizontal=True):
        st.button("New questions", icon=":material/refresh:", type="primary",
                  on_click=lambda: st.session_state.update(quiz_regen=True))
        if missed and answered == len(qs):
            st.button(f"Retry the {missed} I missed", icon=":material/replay:",
                      on_click=retry_quiz, args=(True,))
        st.button("Try again", icon=":material/restart_alt:", on_click=retry_quiz)
        st.download_button("Download study sheet", study_sheet(quiz),
                           file_name="practice_questions.md", mime="text/markdown",
                           icon=":material/download:", on_click="ignore")


def page():
    lectures = dict(c.rag().lectures)
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
