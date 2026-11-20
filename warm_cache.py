"""Prepare answers for the starter questions and the demo path, and a summary of every demo
lecture in each interface language, so they appear instantly on the website and still work if
the AI service is busy.  Run:  python warm_cache.py          (both)
                               python warm_cache.py --summaries   (summaries only)
Re-run after rebuilding the index (old entries are ignored automatically once it changes).
Uses one AI request per question (two for questions not in English, which are translated
first) and one per summary; anything saved already is skipped."""
import argparse
import time

import answer_cache
import config
from rag import SlideRAG, citation_facts, refusal_for

# Asked during the demo; the starter chips come from config.DEMO_QUESTIONS.
EXTRA = [
    "What is chunk overlap in Lesson 3?",
    "What must students submit for the lab?",
    "What does reciprocal rank measure?",
    "What are the key takeaways of Lesson 2?",
    "RAG 시스템의 실패는 어떻게 진단하나요?",
]


def warm_answers(rag: SlideRAG):
    answers = answer_cache.load_file()
    questions = list(dict.fromkeys(list(config.DEMO_QUESTIONS.values()) + EXTRA
                                   + list(config.DEMO_QUESTIONS_RU.values())
                                   + list(config.DEMO_QUESTIONS_KO.values())))
    for q in questions:
        key = answer_cache.cache_key(q, config.TOP_K)
        if key in answers:
            print(f"cached already: {q}")
            continue
        t0 = time.time()
        info = {}
        messages, hits = rag.prepare(q, config.TOP_K, info=info)
        trace = {"n_slides": len(rag.chunks), "n_lectures": len(rag.lectures), "k": config.TOP_K,
                 "search_s": time.time() - t0, **info}
        if messages is None:
            answer = refusal_for(q)
            trace["refused_early"] = True
        else:
            t1 = time.time()
            answer = rag.chat(messages, info)
            if answer is None:
                print(f"BUSY, skipped: {q}")
                continue
            trace.update(answer_s=time.time() - t1, model=info["model"])
        answers[key] = answer_cache.to_entry(answer, hits, trace)
        print(f"saved ({trace.get('model')}): {q}\n   {answer[:120]!r}")
    answer_cache.save_file(answers)
    print(f"{len(answers)} answers in {answer_cache.CACHE_FILE}")


def warm_summaries(rag: SlideRAG):
    """Every lecture x interface language. Prints how many slides each summary cites and how
    many of its citations don't match a slide of that lecture; a summary with any of those is
    written again (up to 3 tries), and not saved if it still has them."""
    summaries = answer_cache.load_summaries()
    for tag, _ in rag.lectures:
        slides = [c for c in rag.chunks if c["lecture"] == tag]
        for lang in config.SUMMARY_HEADINGS:
            key = answer_cache.summary_key(tag, lang)
            if key in summaries:
                print(f"cached already: {key}")
                continue
            for attempt in range(3):  # again if a citation doesn't match a slide of the lecture
                info, t0 = {}, time.time()
                text = "".join(rag.summarize(tag, lang, info))
                if info.get("model") is None or "cut off" in text:
                    break
                facts = citation_facts(text, slides)
                print(f"{key} ({info['model']}, {time.time() - t0:.1f} s): {facts['words']} "
                      f"words, cites {facts['cited']} of {len(slides)} slides, "
                      f"{facts['made_up']} not in the lecture")
                if not facts["made_up"]:
                    break
            if info.get("model") is None or "cut off" in text or facts["made_up"]:
                print(f"   NOT saved: {key}")
                continue
            summaries[key] = {"text": text, "model": info["model"],
                              "seconds": round(time.time() - t0, 1), "slides": info["slides"],
                              "trimmed": info["trimmed"]}
            answer_cache.save_summaries(summaries)  # after each one, so a stop loses nothing
    print(f"{len(summaries)} summaries in {answer_cache.SUMMARY_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summaries", action="store_true", help="only the lecture summaries")
    args = parser.parse_args()
    rag = SlideRAG()
    if not args.summaries:
        warm_answers(rag)
    warm_summaries(rag)


if __name__ == "__main__":
    main()
