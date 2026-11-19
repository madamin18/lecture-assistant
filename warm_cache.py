"""Prepare answers for the starter questions and the demo path, so they appear instantly on the
website and still work if the AI service is busy.  Run:  python warm_cache.py
Re-run after rebuilding the index (old answers are ignored automatically once it changes).
Uses one AI request per question (two for Korean questions, which are translated first)."""
import time

import answer_cache
import config
from rag import SlideRAG

# Asked during the demo; the starter chips come from config.DEMO_QUESTIONS.
EXTRA = [
    "What is chunk overlap in Lesson 3?",
    "What must students submit for the lab?",
    "What does reciprocal rank measure?",
    "What are the key takeaways of Lesson 2?",
    "RAG 시스템의 실패는 어떻게 진단하나요?",
]


def main():
    rag = SlideRAG()
    answers = answer_cache.load_file()
    questions = list(config.DEMO_QUESTIONS.values()) + EXTRA
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
            from rag import refusal_for
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


if __name__ == "__main__":
    main()
