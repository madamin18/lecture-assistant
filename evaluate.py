"""Step 4: evaluation for the 'Testing and improvements' part of the rubric.

Test set format (eval/test_set.jsonl), one JSON object per line:
  {"id": "q01", "type": "direct", "question": "...", "expected": ["L03:14"], "reference": "..."}
  expected = list of "LECTURE:PAGE" that contain the answer; [] means out-of-scope (should refuse).

Run:  python evaluate.py --name baseline
      python evaluate.py --name hybrid --judge     (also asks the LLM to grade answers)
Results go to eval/results_<name>.jsonl and eval/summary_<name>.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import config
from rag import SlideRAG

CITE_RE = re.compile(r"([A-Za-z0-9_]+) p\.(\d+)(?:-(\d+))?")


def parse_expected(items):
    out = []
    for s in items:
        lec, page = s.split(":")
        out.append((lec.upper(), int(page)))
    return out


def chunk_covers(chunk, lec, page):
    return chunk["lecture"].upper() == lec and page in chunk["pages"]


def cited_slides(answer):
    found = set()
    for lec, start, end in CITE_RE.findall(answer):
        for p in range(int(start), int(end or start) + 1):
            found.add((lec.upper(), p))
    return found


def judge(rag, question, reference, answer):
    prompt = (f"Question: {question}\nReference answer: {reference}\nStudent-facing answer: {answer}\n\n"
              "Is the answer factually consistent with the reference and does it answer the question? "
              "Reply with only CORRECT, PARTIAL, or WRONG.")
    reply = rag.chat([{"role": "user", "content": prompt}])  # same fallback models as answers
    if not reply:
        return None  # every model busy: leave this row unjudged
    return reply.strip().upper().split()[0].strip(".*")


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
    args = ap.parse_args()
    if args.min_score is not None:
        config.MIN_SCORE = args.min_score

    tests = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    rag = SlideRAG()
    rows = []

    for i, t in enumerate(tests):
        if i and args.delay:
            time.sleep(args.delay)
        answer, hits = rag.answer(t["question"], args.k)
        expected = parse_expected(t.get("expected", []))
        refused = config.REFUSAL in answer
        busy = answer == config.BUSY_MESSAGE
        row = {"id": t["id"], "type": t.get("type", ""), "question": t["question"],
               "answer": answer, "retrieved": [h["cite"] for h in hits],
               "top_score": round(hits[0]["score"], 4) if hits else None, "refused": refused,
               "busy": busy, "model": rag.last_model}  # None = LLM not called (threshold refusal)

        if busy:  # no real answer: don't count it as right or wrong
            rows.append(row)
            print(f"{t['id']}: BUSY - every model failed, skipped in the scores")
            continue
        if expected:  # in-scope question
            row["retrieval_hit"] = any(chunk_covers(h, l, p) for h in hits for l, p in expected)
            row["citation_correct"] = bool(cited_slides(answer) & set(expected))
            if args.judge and t.get("reference"):
                row["judge"] = judge(rag, t["question"], t["reference"], answer)
        else:         # out-of-scope question
            row["correct_refusal"] = refused

        rows.append(row)
        print(f"{t['id']}: refused={refused} hit={row.get('retrieval_hit', '-')}")

    ins = [r for r in rows if "retrieval_hit" in r]
    outs = [r for r in rows if "correct_refusal" in r]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    summary = {
        "name": args.name, "k": args.k, "min_score": config.MIN_SCORE,
        "n_in_scope": len(ins), "n_out_of_scope": len(outs),
        "n_busy_skipped": sum(r["busy"] for r in rows),
        f"retrieval_hit@{args.k}_%": pct([r["retrieval_hit"] for r in ins]),
        "citation_accuracy_%": pct([r["citation_correct"] for r in ins]),
        "false_refusal_%": pct([r["refused"] for r in ins]),
        "correct_refusal_%": pct([r["correct_refusal"] for r in outs]),
    }
    if args.judge:
        judged = [r["judge"] for r in ins if r.get("judge")]
        summary["answer_correct_%"] = pct([j == "CORRECT" for j in judged])
        summary["answer_partial_%"] = pct([j == "PARTIAL" for j in judged])

    Path("eval").mkdir(exist_ok=True)
    with open(f"eval/results_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(f"eval/summary_{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Helps pick MIN_SCORE: a good threshold sits between these two ranges.
    rng = lambda xs: f"{min(xs):.3f} - {max(xs):.3f}" if xs else "n/a"
    print(f"\ntop_score  in-scope:     {rng([r['top_score'] for r in ins])}")
    print(f"top_score  out-of-scope: {rng([r['top_score'] for r in outs])}")


if __name__ == "__main__":
    main()
