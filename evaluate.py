"""Step 4: evaluation for the 'Testing and improvements' part of the rubric.

Test set format (eval/test_set.jsonl), one JSON object per line:
  {"id": "q01", "type": "direct", "question": "...", "expected": ["L03:14"], "reference": "..."}
  expected = list of "LECTURE:PAGE" that contain the answer; [] means out-of-scope (should refuse).
  PAGE is the page's position in the PDF (what a PDF viewer shows), not the number printed
  on the slide, so the test set doesn't change if slide numbering is detected differently.

Run:  python evaluate.py --name baseline
      python evaluate.py --name hybrid --judge     (also asks the LLM to grade answers)
      python evaluate.py --retrieval-only          (search only; the API is used only to
                                                    translate cross-language questions)
      python evaluate.py --retrieval-only --no-hybrid --no-translate   (older search, no API)
Results go to eval/results_<name>.jsonl and eval/summary_<name>.json
"""
import argparse
import json
import time
from pathlib import Path

import config
from rag import SlideRAG, cited_hits, language


def parse_expected(items):
    out = []
    for s in items:
        lec, page = s.split(":")
        out.append((lec.upper(), int(page)))
    return out


def chunk_covers(chunk, lec, page):
    return chunk["lecture"].upper() == lec and page in chunk["pages"]


def first_rank(hits, expected):
    """1-based position of the first retrieved slide that contains the answer (None = missed)."""
    for rank, h in enumerate(hits, start=1):
        if any(chunk_covers(h, l, p) for l, p in expected):
            return rank
    return None


def mrr(ranks):
    """Mean reciprocal rank: 1.0 = the right slide is always first."""
    return round(sum(1 / r for r in ranks if r) / len(ranks), 3) if ranks else None


pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None


def load_tests(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def retrieval_only(args):
    rag = SlideRAG()
    ranks = {"en": [], "ko": []}
    for t in load_tests(args.test):
        expected = parse_expected(t.get("expected", []))
        if not expected:
            continue
        hits, _ = rag.search(t["question"], args.k)
        rank = first_rank(hits, expected)
        ranks[language(t["question"])].append(rank)
        print(f"{t['id']}: {'rank ' + str(rank) if rank else 'MISSED'}")
    allr = ranks["en"] + ranks["ko"]
    out = {"hybrid_search": config.HYBRID_SEARCH, "translate_queries": config.TRANSLATE_QUERIES,
           "k": args.k, f"retrieval_hit@{args.k}_%": pct([r is not None for r in allr]),
           "retrieval_mrr": mrr(allr)}
    for lang, rs in ranks.items():
        if rs:
            out[f"{lang}: hit@{args.k}_% (n={len(rs)})"] = pct([r is not None for r in rs])
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cited_slides(answer, rag):
    """(LECTURE, PDF page) for every slide the answer cites."""
    return {(c["lecture"].upper(), p) for c in cited_hits(answer, rag.chunks) for p in c["pages"]}


def judge(rag, question, reference, answer):
    prompt = (f"Question: {question}\nReference answer: {reference}\nStudent-facing answer: {answer}\n\n"
              "Is the answer factually consistent with the reference and does it answer the question? "
              "The answer may be in a different language from the reference. "
              "Reply with only CORRECT, PARTIAL, or WRONG.")
    reply = rag.chat([{"role": "user", "content": prompt}])  # same fallback models as answers
    if not reply:
        return None  # every model busy: leave this row unjudged
    return reply.strip().upper().split()[0].strip(".*")


def scores(rows, k, judged_run):
    ins = [r for r in rows if "retrieval_hit" in r]
    outs = [r for r in rows if "correct_refusal" in r]
    s = {
        "n_in_scope": len(ins), "n_out_of_scope": len(outs),
        f"retrieval_hit@{k}_%": pct([r["retrieval_hit"] for r in ins]),
        "retrieval_mrr": mrr([r["rank"] for r in ins]),
        "citation_accuracy_%": pct([r["citation_correct"] for r in ins]),
        "false_refusal_%": pct([r["refused"] for r in ins]),
        "correct_refusal_%": pct([r["correct_refusal"] for r in outs]),
    }
    if judged_run:
        judged = [r["judge"] for r in ins if r.get("judge")]
        s["answer_correct_%"] = pct([j == "CORRECT" for j in judged])
        s["answer_partial_%"] = pct([j == "PARTIAL" for j in judged])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--test", default="eval/test_set.jsonl")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--min-score", type=float, default=None,
                    help="override config.MIN_SCORE for this run (refusal threshold)")
    ap.add_argument("--delay", type=float, default=0,
                    help="seconds to wait between questions (free-tier rate limits)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="only test the search, no answers")
    ap.add_argument("--no-hybrid", action="store_true",
                    help="turn off keyword search (config.HYBRID_SEARCH) for this run")
    ap.add_argument("--no-translate", action="store_true",
                    help="don't translate cross-language questions (config.TRANSLATE_QUERIES)")
    args = ap.parse_args()
    if args.min_score is not None:
        config.MIN_SCORE = args.min_score
    if args.no_hybrid:
        config.HYBRID_SEARCH = False
    if args.no_translate:
        config.TRANSLATE_QUERIES = False
    if args.retrieval_only:
        return retrieval_only(args)

    tests = load_tests(args.test)
    rag = SlideRAG()
    rows = []

    for i, t in enumerate(tests):
        if i and args.delay:
            time.sleep(args.delay)
        answer, hits = rag.answer(t["question"], args.k)
        expected = parse_expected(t.get("expected", []))
        refused = config.REFUSAL in answer or config.REFUSAL_KO in answer
        busy = answer == config.BUSY_MESSAGE
        row = {"id": t["id"], "type": t.get("type", ""), "lang": language(t["question"]),
               "question": t["question"], "answer": answer, "retrieved": [h["cite"] for h in hits],
               "top_score": round(max(h["score"] for h in hits), 4) if hits else None,
               "refused": refused,
               "busy": busy, "model": rag.last_model}  # None = LLM not called (threshold refusal)

        if busy:  # no real answer: don't count it as right or wrong
            rows.append(row)
            print(f"{t['id']}: BUSY - every model failed, skipped in the scores")
            continue
        if expected:  # in-scope question
            row["retrieval_hit"] = any(chunk_covers(h, l, p) for h in hits for l, p in expected)
            row["rank"] = first_rank(hits, expected)
            row["citation_correct"] = bool(cited_slides(answer, rag) & set(expected))
            if args.judge and t.get("reference"):
                row["judge"] = judge(rag, t["question"], t["reference"], answer)
        else:         # out-of-scope question
            row["correct_refusal"] = refused

        rows.append(row)
        print(f"{t['id']}: refused={refused} hit={row.get('retrieval_hit', '-')}")

    summary = {
        "name": args.name, "k": args.k, "min_score": config.MIN_SCORE,
        "model": config.LLM_MODEL, "hybrid_search": config.HYBRID_SEARCH,
        "translate_queries": config.TRANSLATE_QUERIES,
        "n_busy_skipped": sum(r["busy"] for r in rows),
        **scores(rows, args.k, args.judge),
        "by_language": {lang: scores([r for r in rows if r["lang"] == lang], args.k, args.judge)
                        for lang in ("en", "ko") if any(r["lang"] == lang for r in rows)},
    }

    Path("eval").mkdir(exist_ok=True)
    with open(f"eval/results_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(f"eval/summary_{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Helps pick MIN_SCORE: a good threshold sits between these two ranges.
    ins = [r for r in rows if "retrieval_hit" in r]
    outs = [r for r in rows if "correct_refusal" in r]
    rng = lambda xs: f"{min(xs):.3f} - {max(xs):.3f}" if xs else "n/a"
    print(f"\ntop_score  in-scope:     {rng([r['top_score'] for r in ins])}")
    print(f"top_score  out-of-scope: {rng([r['top_score'] for r in outs])}")


if __name__ == "__main__":
    main()
