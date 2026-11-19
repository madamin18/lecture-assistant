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
from ui import resources

LETTERS = "ABC"
EXAMPLES = ["What is a hallucination?", "How is recall@k calculated?",
            "Why does chunk overlap help retrieval?", "에이전트와 워크플로의 차이는 무엇인가요?"]
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


def use_example():
    st.session_state.compare_q = st.session_state.compare_example
    st.session_state.compare_example = None


def run(question: str, picked: list, blind: bool):
    rag = c.rag()
    info = {}
    with st.spinner("Searching the slides…"):
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
            slot.info(f"{label(cmp, i, order[i])} is writing…", icon=":material/hourglass_top:")
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
    return f"Model {LETTERS[i]}" if hidden else rag_module.model_label(ref)


def show_answer(cmp: dict, i: int):
    r = cmp["results"][i]
    won = cmp["vote"] == i
    st.markdown(f"**{label(cmp, i, r['model'])}**" + (" :green-badge[:material/thumb_up: your pick]"
                                                      if won else ""))
    if r["text"] is None:
        st.warning(f"No answer: {r['error'] or 'busy'}.", icon=":material/error:")
        return
    f = rag_module.citation_facts(r["text"], cmp["hits"])
    badges = [f":gray-badge[:material/timer: {r['seconds']:.1f} s]",
              f":violet-badge[{f['cited']} slide{'s' * (f['cited'] != 1)} cited]"]
    if f["made_up"]:
        badges.append(f":red-badge[{f['made_up']} made-up citation{'s' * (f['made_up'] > 1)}]")
    if f["refused"]:
        badges.append(":orange-badge[refused]")
    st.markdown(" ".join(badges))
    st.markdown(rag_module.badge_citations(r["text"]))


def vote(verdict):
    cmp = st.session_state.comparison
    cmp["vote"] = verdict
    resources.record_vote(cmp["results"], verdict)
    st.toast("Thanks! Your vote is in the leaderboard below.", icon=":material/how_to_vote:")


def show_comparison(cmp: dict):
    st.markdown(f"#### :material/help: {cmp['question']}")
    if not cmp["results"]:
        st.info("No slide is close enough to this question, so every model would get the same "
                "refusal. Try a question about the lectures.", icon=":material/search_off:")
        return
    with st.container(horizontal=True, gap="medium"):
        for i in range(len(cmp["results"])):
            with st.container(border=True, width="stretch"):
                show_answer(cmp, i)

    answered = [i for i, r in enumerate(cmp["results"]) if r["text"] is not None]
    if cmp["vote"] is None and len(answered) >= 2:
        st.markdown("**Which answer is better?** Check the cited slides below before you vote.")
        with st.container(horizontal=True):
            for i in answered:
                st.button(f"{label(cmp, i, cmp['results'][i]['model'])} is better",
                          key=f"vote_{cmp['id']}_{i}", on_click=vote, args=(i,),
                          icon=":material/thumb_up:", type="primary")
            st.button("Tie", key=f"vote_{cmp['id']}_tie", on_click=vote, args=("tie",),
                      icon=":material/balance:")
            st.button("Both bad" if len(answered) == 2 else "All bad",
                      key=f"vote_{cmp['id']}_bad", on_click=vote, args=("bad",),
                      icon=":material/thumb_down:")
    elif cmp["vote"] is not None:
        v = cmp["vote"]
        verdict = ({"tie": "a tie", "bad": "no good answer"}[v] if isinstance(v, str)
                   else rag_module.model_label(cmp["results"][v]["model"]))
        st.success(f"You voted for {verdict}.", icon=":material/how_to_vote:")

    with st.expander(f"Slides every model read ({len(cmp['hits'])})", icon=":material/slideshow:"):
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
    st.markdown("##### :material/leaderboard: Visitors' votes")
    if not any(r["votes"] for r in rows.values()):
        st.caption("No votes yet. Compare two models above and vote for the better answer.")
    else:
        table = pd.DataFrame([{
            "Model": rag_module.model_label(m).split(" · ", 1)[-1],
            "Win rate": (r["wins"] + 0.5 * r["ties"]) / r["votes"],
            "Votes": r["votes"], "Ties": r["ties"], "Both bad": r["bad"],
            "Avg time (s)": round(r["seconds"] / r["votes"], 1),
            "No answer": r["failures"],
        } for m, r in rows.items() if r["votes"]]).sort_values("Win rate", ascending=False)
        st.dataframe(table, hide_index=True, column_config={
            "Win rate": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent",
                help="Wins, plus half of each tie, out of the comparisons it answered")})
        st.caption("Votes from everyone using this site since the server last restarted. "
                   "A tie counts as half a win.")

    if BENCHMARK.exists():
        data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        st.markdown("##### :material/science: Measured on the test set")
        st.dataframe(pd.DataFrame([{
            "Model": rag_module.model_label(m["model"]),
            "Correct": m.get("answer_correct_%"), "Partly correct": m.get("answer_partial_%"),
            "Cites a right slide": m.get("citation_accuracy_%"),
            "Made-up citations": m.get("made_up_citations"),
            "Wrongly refused": m.get("false_refusal_%"),
            "Refused off-topic": m.get("correct_refusal_%"),
            "Avg time (s)": m.get("avg_seconds"),
        } for m in data["models"]]), hide_index=True)
        st.caption(f"{data['n_questions']} questions from eval/test_set.jsonl ({data['date']}); "
                   f"percentages. Answers graded by {rag_module.model_label(data['judge'])}, the "
                   "same judge for every model.")


def page():
    models = resources.available_models(c.version())
    st.subheader(":material/compare_arrows: Which AI model answers best?", anchor=False)
    st.caption("Every model gets the same question and the same slides, so only the writing "
               "differs. Vote for the better answer; the names stay hidden until you vote.")
    if len(models) < 2:
        st.info("Only one AI model is available. Add another provider's API key (e.g. "
                "GROQ_API_KEY) to compare models.", icon=":material/key:")
        return

    st.pills("Examples", EXAMPLES, key="compare_example", on_change=use_example,
             label_visibility="collapsed")
    with st.form("compare_form"):
        picked = st.multiselect("Models (2 or 3)", models, default=default_models(models),
                                max_selections=3, format_func=rag_module.model_label,
                                key="compare_models")
        question = st.text_input("Question", key="compare_q",
                                 placeholder="Ask about the lectures, in English or Korean")
        blind = st.toggle("Hide model names until I vote", value=True, key="compare_blind")
        go = st.form_submit_button("Compare answers", type="primary", width="stretch",
                                   icon=":material/play_arrow:")
    if go:
        if len(picked) < 2:
            st.warning("Pick at least two models.", icon=":material/warning:")
        elif not question.strip():
            st.warning("Type a question first.", icon=":material/warning:")
        else:
            run(question.strip(), picked, blind)

    if st.session_state.get("comparison"):
        show_comparison(st.session_state.comparison)
    leaderboard()
