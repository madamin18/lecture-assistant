"""Pure text helpers in rag.py: keywords, language, lecture mentions, citations."""
import numpy as np

import config
from rag import (BM25, badge_citations, citation_facts, cited_hits, language, mentioned_lectures,
                 refusal_for, refused, retry_delay, rrf_ranks, tokenize)
from conftest import status_error


def test_tokenize_drops_stopwords_and_single_letters():
    assert tokenize("What is the RAG pipeline, a b?") == ["rag", "pipeline"]


def test_tokenize_korean_pieces_match_inflected_words():
    # '프롬프트의' (with a particle) still shares 2-character pieces with '프롬프트'
    assert {"프롬", "롬프", "프트"} <= set(tokenize("프롬프트의"))


def test_language():
    assert language("What is a hallucination?") == "en"
    assert language("환각이란 무엇인가요?") == "ko"
    assert language("RAG란 무엇인가요?") == "ko"
    assert language("12345") == "en"


def test_mentioned_lectures():
    available = {"L01", "L03", "L05"}
    assert mentioned_lectures("What must students submit in the Lesson 1 lab?", available) == {"L01"}
    assert mentioned_lectures("lecture 03 and L05", available) == {"L03", "L05"}
    assert mentioned_lectures("3강에서 청크란?", available) == {"L03"}
    assert mentioned_lectures("5차시 내용", available) == {"L05"}
    assert mentioned_lectures("Что нужно сдать по лекции 1?", available) == {"L01"}
    assert mentioned_lectures("урок 3: чанки", available) == {"L03"}
    assert mentioned_lectures("Is the L2 norm used?", available) == set()  # not lecture 2
    assert mentioned_lectures("Lesson 7", available) == set()  # no such lecture


def test_bm25_prefers_rare_matching_word():
    docs = [tokenize(t) for t in ("submit evidence folder", "evidence evidence", "unrelated text")]
    s = BM25(docs).scores(tokenize("what must I submit"))
    assert s.argmax() == 0 and s[2] == 0
    assert not BM25(docs).scores(["nowhere"]).any()


def test_rrf_only_positive_ignores_zero_scores():
    contrib = rrf_ranks(np.array([0.0, 2.0, 1.0]), only_positive=True)
    assert contrib[1] == 1 / 61 and contrib[2] == 1 / 62 and contrib[0] == 0


def test_retry_delay_reads_gemini_429_body():
    assert retry_delay(status_error(429, '{"retryDelay": "76943s"}')) == 76943
    assert retry_delay(status_error(429, "no delay")) is None


def test_retry_delay_reads_groq_messages_and_headers():
    groq = "Rate limit reached ... tokens per day (TPD): Limit 200000. Please try again in {}."
    assert retry_delay(status_error(429, groq.format("3m36.5s"))) == 216.5
    assert retry_delay(status_error(429, groq.format("1h2m"))) == 3720
    assert retry_delay(status_error(429, groq.format("7.66s"))) == 7.66
    assert retry_delay(status_error(429, groq.format("860ms"))) == 0.86
    assert retry_delay(status_error(429, "slow down", headers={"retry-after": "30"})) == 30
    assert retry_delay(status_error(429, "Please try again in a minute.")) is None


def test_cited_hits_does_not_match_prefix():
    hits = [{"cite": "L01 p.1"}, {"cite": "L01 p.13"}, {"cite": "L02 slide 4"}]
    assert cited_hits("See [L01 p.13] and [L02 slide 4].", hits) == hits[1:]


def test_citation_formats_from_other_models():
    hits = [{"cite": "L01 slide 6"}, {"cite": "L01 slide 30"}]
    # gpt-oss puts spaces inside the brackets; Gemini sometimes zero-pads the number
    answer = "Tone is not evidence. [ L01 slide 30 ] Fluency hides gaps [L01 slide 06]."
    assert cited_hits(answer, hits) == hits
    facts = citation_facts(answer, hits)
    assert (facts["cited"], facts["made_up"], facts["refused"]) == (2, 0, False)
    assert badge_citations("[ L01 slide 30 ]") == ":violet-badge[L01 slide 30]"
    nbsp = "Tone [ L01 slide 30 ]"  # gpt-oss's narrow no-break spaces
    assert cited_hits(nbsp, hits) == hits[1:]
    assert badge_citations(nbsp) == "Tone :violet-badge[L01 slide 30]"
    assert citation_facts("Made up [L09 slide 99].", hits)["made_up"] == 1


def test_russian_questions():
    assert language("Что такое галлюцинация?") == "ru"
    assert refusal_for("Где находится столовая?") == config.REFUSAL_RU
    assert refused(config.REFUSAL_RU) and not refused("RAG uses retrieval.")


def test_badge_citations():
    assert badge_citations("RAG grounds answers [L03 slide 14].") == \
        "RAG grounds answers :violet-badge[L03 slide 14]."
    assert badge_citations("[L01 slide 3; L01 slide 9]") == \
        ":violet-badge[L01 slide 3] :violet-badge[L01 slide 9]"
    assert badge_citations("[L02 p.5-6]") == ":violet-badge[L02 p.5-6]"
    untouched = "A [link](https://example.com) and [a note] stay as they are."
    assert badge_citations(untouched) == untouched
