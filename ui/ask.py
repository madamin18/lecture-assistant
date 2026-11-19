"""Ask tab: chat with the slides. Each answer shows how it was found, the slides it came from,
and follow-up suggestions."""
import re
import time

import streamlit as st
from openai import APIStatusError

import answer_cache
import config
import rag as rag_module
from ui import common as c
from ui import i18n, resources
from ui.i18n import t

# Slides that make poor follow-up questions.
SKIP_TITLES = re.compile(r"^(learning outcomes|references?|sources|exit questions|agenda|"
                         r"outline|roadmap|questions|small quizzes)", re.IGNORECASE)


def next_questions(answer: str, hits: list) -> dict:
    """Up to 3 follow-ups built from the other slides the search found (no extra AI call)."""
    cited = rag_module.cited_hits(answer, hits)
    pool = [h for h in hits if h not in cited] or hits
    out = {}
    for h in pool:
        title = c.heading(h).rstrip("…")
        if (1 in h["pages"] or SKIP_TITLES.match(title) or len(title) < 12
                or any(title in v for v in out.values())):
            continue
        out[f":material/arrow_forward: {h['lecture']} · {c.shorten(title, 60)}"] = \
            t("explain_from", title=title, lec=h["lecture"])
        if len(out) == 3:
            break
    return out


def pick_chip(widget_key: str, options: dict):
    """Turn a clicked chip into the next question, then unselect it."""
    st.session_state.pending_question = options[st.session_state[widget_key]]
    st.session_state[widget_key] = None


def starter_chips() -> dict:
    if not c.uploaded():  # answers prepared by warm_cache.py
        return config.DEMO_QUESTIONS_RU if i18n.lang() == "ru" else config.DEMO_QUESTIONS
    return {t("takeaways_label", tag=tag): t("takeaways_q", tag=tag)
            for tag, _ in c.rag().lectures[:4]}


def show_history(messages: list):
    for i, m in enumerate(messages):
        with st.chat_message(m["role"]):
            if m["role"] != "assistant":
                st.markdown(m["content"])
                continue
            c.show_answer_text(m["content"])
            if m.get("trace"):
                with st.expander(c.trace_label(m["trace"]), type="compact",
                                 icon=":material/manage_search:"):
                    c.trace_steps(m["trace"])
            if m.get("hits"):
                c.show_sources(m["content"], m["hits"])
            if i == len(messages) - 1 and m.get("hits"):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.feedback("thumbs", key=f"fb_{len(messages)}", on_change=c.thanks)
                options = next_questions(m["content"], m["hits"])
                if options:
                    st.pills(t("ask_next"), list(options), key=f"next_{len(messages)}",
                             on_change=pick_chip, args=(f"next_{len(messages)}", options))


def welcome():
    with st.container(border=True):
        st.subheader(t("welcome_title"), anchor=False)
        st.markdown(t("welcome_text"))
        st.caption(t("welcome_tip"))
        options = starter_chips()
        st.pills(t("try_asking"), list(options), key=f"starter_{i18n.lang()}",
                 on_change=pick_chip, args=(f"starter_{i18n.lang()}", options))


def cached_answer(question: str, ckey: str, model):
    """A saved answer for a chat's first question, or None."""
    entry = resources.runtime_answers(c.version()).get(ckey)
    if entry is None and not c.uploaded() and model is None:  # saved answers come from Auto
        entry = resources.saved_answers(c.version()).get(
            answer_cache.cache_key(question, c.top_k(), c.lecture_filter()))
    return answer_cache.from_entry(entry, c.rag().chunks) if entry else None


def answer_live(question: str, history: list, model):
    """Search, then stream the answer. Returns (answer, hits, trace), or None after an error."""
    rag, k = c.rag(), c.top_k()
    trace = {"n_slides": len(rag.chunks), "n_lectures": len(rag.lectures), "k": k}
    info = {}  # filled in by the search and the answer (translation, lectures, model)
    try:
        t0 = time.time()
        with st.status(t("searching"), type="compact") as status:
            prepared, hits = rag.prepare(question, k, history, c.lecture_filter(), info)
            trace.update(search_s=time.time() - t0, **info)
            c.trace_steps({**trace, "model": None})
            status.update(label=c.trace_label(trace), state="complete")
        if prepared is None:  # nothing close enough in the slides: refuse without the LLM
            answer = rag_module.refusal_for(question)
            trace["refused_early"] = True
            st.markdown(answer)
        else:
            t1 = time.time()
            answer, box = "", st.empty()
            for piece in rag.stream(prepared, info, model=model):  # appears as it's written
                answer += piece
                box.markdown(rag_module.badge_citations(answer) + " ▌")
            with box.container():
                c.show_answer_text(answer)
            trace.update(answer_s=time.time() - t1, model=info.get("model"),
                         failed=info.get("failed"))
    except APIStatusError as e:
        c.api_error(e)
        return None
    return answer, hits, trace


def page():
    messages = st.session_state.messages
    show_history(messages)

    question = (st.chat_input(t("chat_placeholder"), submit_mode="disable")
                or st.session_state.pop("pending_question", None))

    if not messages:
        box = st.empty()
        with box.container():
            welcome()
        if question:
            box.empty()  # hide the welcome card before the first answer renders
    elif st.session_state.lecture_filter:
        st.caption(t("filter_caption", lectures=", ".join(st.session_state.lecture_filter)))

    if not question:
        return
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    with st.chat_message("user"):
        st.markdown(question)

    # A chat's first question may have been answered before: show that instantly.
    model = c.chosen_model()
    ckey = (f"{c.corpus()}|{model or c.AUTO}|"
            f"{answer_cache.cache_key(question, c.top_k(), c.lecture_filter())}")
    found = cached_answer(question, ckey, model) if not messages else None
    if found is None:
        with st.chat_message("assistant"):
            found = answer_live(question, history, model)
        if found is None:
            return
        answer, hits, trace = found
        if (not messages and answer.strip() and config.BUSY_MESSAGE not in answer
                and "cut off" not in answer):
            resources.remember_answer(c.version(), ckey, answer_cache.to_entry(answer, hits, trace))
    answer, hits, trace = found
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "hits": hits, "trace": trace}]
    st.rerun()  # redraw with slides, feedback and follow-up suggestions
