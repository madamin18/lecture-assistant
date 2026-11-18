"""Evaluate the practice-question generator.  Run:  python eval_quiz.py [--delay 3]

Generates a 5-question quiz for every lecture plus one in Korean, then checks every question
against the slide it cites with a separate LLM call (temperature 0):
  answerable  can the question be answered from that slide alone?
  key_correct is the marked answer (or model answer) correct according to the slide?
  one_correct multiple choice only: is exactly one option correct?
  clear       is the question unambiguous and well formed?
Writes eval/quiz_results.jsonl (every question) and eval/quiz_quality.json (summary).
About 60 AI requests. The checker is the same model family as the generator, so treat the
numbers as an estimate and read eval/quiz_results.jsonl for the flagged questions."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rag import SlideRAG

CHECK_PROMPT = """You are checking an exam practice question written from a lecture slide.
Use ONLY the slide text. Reply with JSON only:
{{"answerable": true/false, "key_correct": true/false, "one_correct": true/false/null,
  "clear": true/false, "issue": "one short sentence, or empty"}}
- answerable: the question can be answered from this slide alone.
- key_correct: the marked answer is correct according to the slide.
- one_correct: for multiple choice, exactly one option is correct (null for short answer).
- clear: the question is unambiguous and well formed.

Slide [{cite}]:
{slide}

Question ({kind}): {question}
{options}Marked answer: {answer}"""

LETTERS = "ABCDE"


def check(rag, q: dict) -> dict | None:
    if q["type"] == "mcq":
        options = "".join(f"{LETTERS[i]}) {o}\n" for i, o in enumerate(q["options"]))
        options = "Options:\n" + options
        answer = f"{LETTERS[q['answer']]}) {q['options'][q['answer']]}"
        kind = "multiple choice"
    else:
        options, answer, kind = "", q["answer"], "short answer"
    prompt = CHECK_PROMPT.format(cite=q["hit"]["cite"], slide=q["hit"]["text"], kind=kind,
                                 question=q["question"], options=options, answer=answer)
    reply = rag.chat([{"role": "user", "content": prompt}], timeout=30,
                     response_format={"type": "json_object"})
    if not reply:
        return None
    try:
        return json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=3, help="seconds between AI requests")
    args = ap.parse_args()
    rag = SlideRAG()
    runs = [([tag], "same") for tag, _ in rag.lectures] + [([], "ko")]
    rows, gen = [], []
    for lectures, lang in runs:
        t0 = time.time()
        qs = rag.generate_quiz(5, "mixed", lectures, "", lang)
        gen.append({"lectures": lectures, "lang": lang, "requested": 5,
                    "delivered": len(qs or []), "seconds": round(time.time() - t0, 1),
                    "model": rag.last_model})
        print(f"quiz {lectures or 'all'} ({lang}): {len(qs or [])} questions in {gen[-1]['seconds']} s")
        for q in qs or []:
            time.sleep(args.delay)
            verdict = check(rag, q)
            rows.append({"lecture": q["hit"]["lecture"], "lang": lang, "type": q["type"],
                         "cite": q["hit"]["cite"], "question": q["question"],
                         "options": q.get("options"), "answer": q["answer"],
                         "verdict": verdict})
            flag = "" if verdict and all(verdict.get(k) in (True, None) for k in
                                         ("answerable", "key_correct", "one_correct", "clear")) \
                else f"  <-- {verdict}"
            print(f"   {q['hit']['cite']:14s} {q['type']:5s} {q['question'][:70]}{flag}")
        time.sleep(args.delay)

    checked = [r for r in rows if r["verdict"]]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    mcqs = [r for r in checked if r["type"] == "mcq"]
    ok = lambda r: all(r["verdict"].get(k) in (True, None)
                       for k in ("answerable", "key_correct", "one_correct", "clear"))
    summary = {
        "quizzes": len(gen), "questions_requested": sum(g["requested"] for g in gen),
        "questions_delivered": len(rows), "questions_checked": len(checked),
        "multiple_choice_share_%": pct([r["type"] == "mcq" for r in rows]),
        "answerable_from_cited_slide_%": pct([r["verdict"].get("answerable") is True for r in checked]),
        "marked_answer_correct_%": pct([r["verdict"].get("key_correct") is True for r in checked]),
        "mcq_exactly_one_correct_%": pct([r["verdict"].get("one_correct") is True for r in mcqs]),
        "clear_%": pct([r["verdict"].get("clear") is True for r in checked]),
        "passed_all_checks_%": pct([ok(r) for r in checked]),
        "correct_option_positions": dict(sorted(Counter(
            LETTERS[r["answer"]] for r in rows if r["type"] == "mcq").items())),
        "avg_generation_seconds": round(sum(g["seconds"] for g in gen) / len(gen), 1),
        "models": dict(Counter(g["model"] for g in gen)),
        "generation_runs": gen,
    }
    Path("eval").mkdir(exist_ok=True)
    with open("eval/quiz_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path("eval/quiz_quality.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "generation_runs"}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
