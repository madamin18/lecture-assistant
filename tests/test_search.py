"""SlideRAG search and prompt building, with the fake embedder and AI client."""
import pytest

import config
from conftest import CHUNKS, FakeClient, chunk, make_rag


@pytest.fixture
def no_threshold(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.0)


def test_retrieve_finds_the_matching_slide():
    hits = make_rag().retrieve("What is chunk overlap?", k=3)
    assert hits[0]["cite"] == "L03 slide 1"
    assert all("score" in h for h in hits)


def test_keyword_search_finds_rare_word():
    hits = make_rag().retrieve("What must students submit?", k=1)
    assert hits[0]["cite"] == "L01 slide 3"


def test_lecture_named_in_question_limits_search():
    info = {}
    hits = make_rag().retrieve("In Lesson 2, what is cosine similarity?", k=3, info=info)
    assert {h["lecture"] for h in hits} == {"L02"}
    assert info == {"lectures": ["L02"], "named": True}


def test_sidebar_filter_and_named_lecture_wins():
    rag = make_rag()
    assert {h["lecture"] for h in rag.retrieve("tokens", k=3, only=["L01"])} == {"L01"}
    hits = rag.retrieve("tokens in lecture 2", k=3, only=["L01"])
    assert {h["lecture"] for h in hits} == {"L02"}


def test_filter_never_pads_with_other_lectures():
    # L03 has one slide; asking for 5 must not fill up with slides from L01/L02
    hits = make_rag().retrieve("overlap", k=5, only=["L03"])
    assert [h["cite"] for h in hits] == ["L03 slide 1"]


def test_search_info_is_per_call():
    rag = make_rag()
    a, b = {}, {}
    rag.search("Lesson 1 hallucination", k=2, info=a)
    rag.search("tokens", k=2, info=b)
    assert a["lectures"] == ["L01"] and b["lectures"] == []  # no state shared between calls
    assert a["translation"] is None


def test_korean_question_is_translated_for_english_slides():
    client = FakeClient(default="What is chunk overlap?")
    info = {}
    hits, _ = make_rag(client=client).search("청크 오버랩이란 무엇인가요?", k=2, info=info)
    assert info["translation"] == "What is chunk overlap?"
    assert hits[0]["cite"] == "L03 slide 1"


def test_prepare_refuses_without_ai_when_nothing_is_close(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.99)
    client = FakeClient()
    messages, hits = make_rag(client=client).prepare("Where is the cafeteria?", k=3)
    assert messages is None and client.calls == []


def test_prepare_builds_prompt_with_tags_and_language(no_threshold):
    history = [{"role": "user", "content": "환각이란?"}, {"role": "assistant", "content": "..."}]
    messages, hits = make_rag().prepare("What is a hallucination?", k=2, history=history)
    assert "[L01 slide 1]" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("(Answer in English.)")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_answer_reports_model_and_refuses_in_korean(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.0)
    info = {}
    text, _ = make_rag(client=FakeClient(default="Answer [L01 slide 1]")).answer(
        "What is a hallucination?", k=2, info=info)
    assert text == "Answer [L01 slide 1]" and info["model"] == "m1"

    monkeypatch.setattr(config, "MIN_SCORE", 1.01)  # nothing is close enough
    info = {}
    text, _ = make_rag(client=FakeClient(default="번역")).answer("식당은 어디인가요?", info=info)
    assert text == config.REFUSAL_KO and info["model"] is None


def test_quiz_slides_cover_the_chosen_lecture(monkeypatch):
    monkeypatch.setattr(config, "MIN_CHARS", 10)  # the test slides are short
    slides = make_rag().quiz_slides(lectures=["L01"], n=2)
    assert slides and {c["lecture"] for c in slides} == {"L01"}


def test_lectures_list_in_index_order():
    rag = make_rag(CHUNKS + [chunk("L04", 1, "Grounding ties every claim to a cited source.")])
    assert [t for t, _ in rag.lectures] == ["L01", "L02", "L03", "L04"]
