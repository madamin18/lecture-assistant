"""Compare models tab: one question goes to 2-3 AI models with the same slides, so only the
writing differs. The visitor votes for the better answer (names hidden until then), and votes
from everyone add up in a leaderboard. The test-set results from compare_models.py are shown
next to it, so opinions can be checked against measurements."""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import APIStatusError

import rag as rag_module
from ui import common as c
from ui import i18n, resources
from ui.i18n import t

LETTERS = "ABC"
BENCHMARK = Path(__file__).resolve().parent.parent / "eval" / "model_comparison.json"


def ask_one(rag, messages: list, ref: str) -> dict:
    """One model's answer (runs in a worker thread: no Streamlit calls here)."""
    t0, info = time.time(), {}
    try:
        text = rag.chat(messages, info, model=ref, fallback=False)
    except APIStatusError as e:
        text, info["error"] = None, rag_module.describe_error(e)
    return {"model": ref, "text": text, "seconds": round(time.time() - t0, 1),
            "error": info.get("error")}


def default_models(models: list) -> list:
    """The main model and the first Auto model of another provider (e.g. Gemini vs Groq's
    gpt-oss-120b): tested models, not whatever comes first alphabetically."""
    first = models[0]
    candidates = [m for m in rag_module.auto_chain() if m in models] + models
    other = next((m for m in candidates
                  if rag_module.split_ref(m)[0] != rag_module.split_ref(first)[0]), models[1])
    return [first, other]


def use_example(widget_key: str):
    st.session_state.compare_q = st.session_state[widget_key]
    st.session_state[widget_key] = None


def run(question: str, picked: list, blind: bool):
    rag = c.rag()
    info = {}
    with st.spinner(t("cmp_searching")):
        messages, hits = rag.prepare(question, c.top_k(), None, c.lecture_filter(), info)
    st.session_state.compare_count = st.session_state.get("compare_count", 0) + 1
    cmp = {"id": st.session_state.compare_count, "question": question, "hits": hits,
           "blind": blind, "vote": None, "results": []}
    if messages is not None:
        order = random.sample(picked, len(picked)) if blind else list(picked)
        cmp["results"] = [None] * len(order)
        cols = st.columns(len(order))
        slots = [col.empty() for col in cols]
        for i, slot in enumerate(slots):
            slot.info(t("cmp_writing", name=label(cmp, i, order[i])),
                      icon=":material/hourglass_top:")
        with ThreadPoolExecutor(max_workers=len(order)) as pool:  # all models at once
            jobs = {pool.submit(ask_one, rag, messages, ref): i for i, ref in enumerate(order)}
            for job in as_completed(jobs):  # show each answer as soon as it arrives
                i = jobs[job]
                cmp["results"][i] = job.result()
                with slots[i].container():
                    show_answer(cmp, i)
    st.session_state.comparison = cmp
    st.rerun()


def label(cmp: dict, i: int, ref: str) -> str:
    hidden = cmp["blind"] and cmp["vote"] is None
    return t("cmp_model_x", letter=LETTERS[i]) if hidden else rag_module.model_label(ref)


def show_answer(cmp: dict, i: int):
    r = cmp["results"][i]
    won = cmp["vote"] == i
    st.markdown(f"**{label(cmp, i, r['model'])}**" + (f" {t('cmp_your_pick')}" if won else ""))
    if r["text"] is None:
        st.warning(t("cmp_no_answer", reason=i18n.reason(r["error"] or "busy")),
                   icon=":material/error:")
        return
    f = rag_module.citation_facts(r["text"], cmp["hits"])
    badges = [f":gray-badge[:material/timer: {t('cmp_secs', s=r['seconds'])}]",
              f":violet-badge[{t('cmp_cited', n=f['cited'])}]"]
    if f["made_up"]:
        badges.append(f":red-badge[{t('cmp_made_up', n=f['made_up'])}]")
    if f["refused"]:
        badges.append(f":orange-badge[{t('cmp_refused')}]")
    st.markdown(" ".join(badges))
    st.markdown(rag_module.badge_citations(r["text"]))


def vote(verdict):
    cmp = st.session_state.comparison
    cmp["vote"] = verdict
    resources.record_vote(cmp["results"], verdict)
    st.toast(t("cmp_thanks"), icon=":material/how_to_vote:")


def show_comparison(cmp: dict):
    st.markdown(f"#### :material/help: {cmp['question']}")
    if not cmp["results"]:
        st.info(t("cmp_refusal_all"), icon=":material/search_off:")
        return
    with st.container(horizontal=True, gap="medium"):
        for i in range(len(cmp["results"])):
            with st.container(border=True, width="stretch"):
                show_answer(cmp, i)

    answered = [i for i, r in enumerate(cmp["results"]) if r["text"] is not None]
    if cmp["vote"] is None and len(answered) >= 2:
        st.markdown(t("cmp_vote_q"))
        with st.container(horizontal=True):
            for i in answered:
                st.button(t("cmp_better", name=label(cmp, i, cmp["results"][i]["model"])),
                          key=f"vote_{cmp['id']}_{i}", on_click=vote, args=(i,),
                          icon=":material/thumb_up:", type="primary")
            st.button(t("cmp_tie"), key=f"vote_{cmp['id']}_tie", on_click=vote, args=("tie",),
                      icon=":material/balance:")
            st.button(t("cmp_both_bad") if len(answered) == 2 else t("cmp_all_bad"),
                      key=f"vote_{cmp['id']}_bad", on_click=vote, args=("bad",),
                      icon=":material/thumb_down:")
    elif cmp["vote"] is not None:
        v = cmp["vote"]
        verdict = (t("cmp_verdict_tie") if v == "tie" else t("cmp_verdict_bad") if v == "bad"
                   else rag_module.model_label(cmp["results"][v]["model"]))
        st.success(t("cmp_voted", verdict=verdict), icon=":material/how_to_vote:")

    with st.expander(t("cmp_slides_read", n=len(cmp["hits"])), icon=":material/slideshow:"):
        cited = {h["cite"] for r in cmp["results"] if r["text"]
                 for h in rag_module.cited_hits(r["text"], cmp["hits"])}
        for h in cmp["hits"]:
            if h["cite"] in cited:
                c.show_slide(h)
        c.slides_table(cmp["hits"])


def leaderboard():
    board = resources.vote_board()
    with board["lock"]:
        rows = {m: dict(v) for m, v in board["models"].items()}
    st.markdown(t("cmp_votes_title"))
    if not any(r["votes"] for r in rows.values()):
        st.caption(t("cmp_no_votes"))
    else:
        win = t("col_win_rate")
        table = pd.DataFrame([{
            t("col_model"): rag_module.model_label(m).split(" · ", 1)[-1],
            win: (r["wins"] + 0.5 * r["ties"]) / r["votes"],
            t("col_votes"): r["votes"], t("col_ties"): r["ties"],
            t("col_both_bad"): r["bad"],
            t("col_avg_time"): round(r["seconds"] / r["votes"], 1),
            t("col_no_answer"): r["failures"],
        } for m, r in rows.items() if r["votes"]]).sort_values(win, ascending=False)
        st.dataframe(table, hide_index=True, column_config={
            win: st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent",
                                                 help=t("win_rate_help"))})
        st.caption(t("cmp_votes_caption"))

    if BENCHMARK.exists():
        data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        st.markdown(t("bench_title"))
        st.dataframe(pd.DataFrame([{
            t("col_model"): rag_module.model_label(m["model"]),
            t("col_correct"): m.get("answer_correct_%"),
            t("col_partial"): m.get("answer_partial_%"),
            t("col_right_slide"): m.get("citation_accuracy_%"),
            t("col_made_up"): m.get("made_up_citations"),
            t("col_false_refusal"): m.get("false_refusal_%"),
            t("col_off_topic"): m.get("correct_refusal_%"),
            t("col_avg_time"): m.get("avg_seconds"),
        } for m in data["models"]]), hide_index=True)
        st.caption(t("bench_caption", n=data["n_questions"], date=data["date"],
                     judge=rag_module.model_label(data["judge"])))


def page():
    models = resources.available_models(c.version())
    st.subheader(t("cmp_title"), anchor=False)
    st.caption(t("cmp_caption"))
    if len(models) < 2:
        st.info(t("cmp_one_model"), icon=":material/key:")
        return

    examples_key = f"compare_example_{i18n.lang()}"
    st.pills("Examples", t("cmp_examples"), key=examples_key, on_change=use_example,
             args=(examples_key,), label_visibility="collapsed")
    with st.form("compare_form"):
        picked = st.multiselect(t("cmp_models"), models, default=default_models(models),
                                max_selections=3, format_func=rag_module.model_label,
                                key="compare_models")
        question = st.text_input(t("cmp_question"), key="compare_q",
                                 placeholder=t("cmp_placeholder"))
        blind = st.toggle(t("cmp_blind"), value=True, key="compare_blind")
        go = st.form_submit_button(t("cmp_go"), type="primary", width="stretch",
                                   icon=":material/play_arrow:")
    if go:
        if len(picked) < 2:
            st.warning(t("cmp_pick2"), icon=":material/warning:")
        elif not question.strip():
            st.warning(t("cmp_need_q"), icon=":material/warning:")
        else:
            run(question.strip(), picked, blind)

    if st.session_state.get("comparison"):
        show_comparison(st.session_state.comparison)
    leaderboard()
