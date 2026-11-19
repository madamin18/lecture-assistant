"""Which AI model writes the best answers? Every model answers the test-set questions from the
same slides (searched once), and one fixed judge grades every answer, so only the model differs.

Run:  python compare_models.py gemini-3.5-flash-lite groq:qwen/qwen3.8-27b groq:openai/gpt-oss-120b
      python compare_models.py --auto            (every model in the Auto chain with a key)
      python compare_models.py ... --judge groq:openai/gpt-oss-120b --limit 10
Writes eval/model_comparison.json (shown in the website's Compare models tab) and
eval/model_answers.jsonl (every answer, for reading).

Cost: one request per question per model, plus one judge request per answered question. Groq's
free tier allows about 4 such requests a minute per model (8,000 tokens/minute), so a run is
paced (--delay) and retries after rate limits.
"""
import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from openai import APIStatusError

import config
import rag as rag_module
from evaluate import cited_slides, load_tests, parse_expected, pct
from rag import SlideRAG, auto_chain, canonical, citation_facts, describe_error, model_label

JUDGE_PROMPT = """Question: {question}
Reference answer: {reference}
Student-facing answer: {answer}

Is the answer factually consistent with the reference and does it answer the question? The
answer may be in a different language from the reference. Reply with only CORRECT, PARTIAL, or
WRONG."""


def ask(rag, messages, ref, delay, tries=4, **kwargs):
    """(text, seconds, error) from exactly this model, waiting out short rate limits."""
    for attempt in range(tries):
        time.sleep(delay)
        info, t0 = {}, time.time()
        try:
            text = rag.chat(messages, info, model=ref, fallback=False, **kwargs)
        except APIStatusError as e:
            return None, 0.0, describe_error(e)
        if text is not None:
            return text, time.time() - t0, None
        wait = rag_module._cooldown_until.get(canonical(ref), 0) - time.time()
        if "rate limit" not in info.get("error", "") or wait > 300:  # daily quota: give up
            return None, 0.0, info.get("error")
        print(f"    {ref}: rate limited, waiting {20 * (attempt + 1)} s")
        time.sleep(20 * (attempt + 1))
    return None, 0.0, "rate limit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help='e.g. gemini-3.5-flash-lite "groq:qwen/qwen3.8-27b"')
    ap.add_argument("--auto", action="store_true", help="compare every model in the Auto chain")
    ap.add_argument("--judge", default=config.LLM_MODEL, help="the one model that grades answers")
    ap.add_argument("--test", default="eval/test_set.jsonl")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (quick try)")
    ap.add_argument("--delay", type=float, default=3, help="seconds before each request")
    args = ap.parse_args()
    models = [canonical(m) for m in (auto_chain() if args.auto else args.models)]
    if len(models) < 1:
        raise SystemExit("Name the models to compare, or use --auto.")
    judge_ref = canonical(args.judge)

    rag = SlideRAG()
    tests = load_tests(args.test)[:args.limit or None]
    rows = {m: [] for m in models}
    out_answers = []
    for n, t in enumerate(tests, 1):
        print(f"[{n}/{len(tests)}] {t['id']}: {t['question'][:60]}")
        expected = parse_expected(t.get("expected", []))
        time.sleep(args.delay if rag_module.language(t["question"]) != rag.slide_lang else 0)
        messages, hits = rag.prepare(t["question"], args.k)  # same slides for every model
        for m in models:
            row = {"id": t["id"], "in_scope": bool(expected)}
            if messages is None:  # nothing close enough: the app refuses without any model
                text, secs, err = rag_module.refusal_for(t["question"]), None, None
            else:
                text, secs, err = ask(rag, messages, m, args.delay)
            row.update(seconds=secs, error=err)
            if text is None:
                print(f"  {model_label(m)}: no answer ({err})")
                rows[m].append(row)
                continue
            f = citation_facts(text, hits)
            row.update(refused=f["refused"], made_up=f["made_up"])
            if expected:
                row["citation_correct"] = bool(cited_slides(text, rag) & set(expected))
                if t.get("reference") and not f["refused"]:
                    prompt = JUDGE_PROMPT.format(question=t["question"],
                                                 reference=t["reference"], answer=text)
                    verdict, _, _ = ask(rag, [{"role": "user", "content": prompt}], judge_ref,
                                        args.delay)
                    row["judge"] = (verdict or "").strip().upper().split()[0].strip(".*") \
                        if verdict else None
                elif f["refused"]:
                    row["judge"] = "WRONG"  # refused an answerable question
            print(f"  {model_label(m)}: {row.get('judge', 'refused' if f['refused'] else '-')}"
                  + (f" in {secs:.1f} s" if secs else ""))
            rows[m].append(row)
            out_answers.append({"id": t["id"], "model": m, "answer": text, **row})

    summary = []
    for m in models:
        rs = [r for r in rows[m] if not r.get("error")]
        ins = [r for r in rs if r["in_scope"]]
        outs = [r for r in rs if not r["in_scope"]]
        judged = [r["judge"] for r in ins if r.get("judge")]
        secs = [r["seconds"] for r in rs if r.get("seconds")]
        summary.append({
            "model": m,
            "answer_correct_%": pct([j == "CORRECT" for j in judged]),
            "answer_partial_%": pct([j == "PARTIAL" for j in judged]),
            "citation_accuracy_%": pct([r["citation_correct"] for r in ins]),
            "made_up_citations": sum(r["made_up"] for r in rs),
            "false_refusal_%": pct([r["refused"] for r in ins]),
            "correct_refusal_%": pct([r["refused"] for r in outs]),
            "avg_seconds": round(statistics.mean(secs), 1) if secs else None,
            "no_answer": len(rows[m]) - len(rs),
        })
    result = {"date": date.today().isoformat(), "judge": judge_ref, "k": args.k,
              "n_questions": len(tests), "models": summary}
    Path("eval").mkdir(exist_ok=True)
    Path("eval/model_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open("eval/model_answers.jsonl", "w", encoding="utf-8") as f:
        for a in out_answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
