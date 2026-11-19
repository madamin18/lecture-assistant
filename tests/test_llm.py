"""Model fallback, cooldowns, streaming, and practice-question parsing."""
import json
import time

import openai
import pytest

import config
import rag
from conftest import CHUNKS, FakeClient, make_rag, status_error, timeout_error
from rag import clean_quiz, find_slide

MESSAGES = [{"role": "user", "content": "hi"}]


def test_falls_back_when_main_model_is_overloaded():
    client = FakeClient({"m1": [status_error(503)]}, default="from m2")
    info = {}
    assert make_rag(client=client).chat(MESSAGES, info) == "from m2"
    assert info["model"] == "m2"
    assert rag._cooldown_until["m1"] > time.time()  # skipped by the next visitor


def test_daily_quota_skips_model_for_up_to_an_hour():
    client = FakeClient({"m1": [status_error(429, '{"retryDelay": "50000s"}')]})
    make_rag(client=client).chat(MESSAGES)
    wait = rag._cooldown_until["m1"] - time.time()
    assert 3500 < wait <= 3600


def test_bad_key_is_not_retried_on_other_models():
    client = FakeClient({"m1": [status_error(401)]})
    with pytest.raises(openai.APIStatusError):
        make_rag(client=client).chat(MESSAGES)
    assert len(client.calls) == 1


def test_empty_reply_tries_next_model():
    client = FakeClient({"m1": [""]}, default="real answer")
    info = {}
    assert make_rag(client=client).chat(MESSAGES, info) == "real answer"
    assert info["model"] == "m2"


def test_all_models_busy_returns_none():
    client = FakeClient({m: [status_error(503)] * 2 for m in ("m1", "m2")})
    assert make_rag(client=client).chat(MESSAGES) is None
    assert len(client.calls) == 4  # every model, then once more after a short wait


def test_stream_falls_back_and_reports_model():
    client = FakeClient({"m1": [timeout_error()], "m2": [["Hello", " world"]]})
    info = {}
    assert "".join(make_rag(client=client).stream(MESSAGES, info)) == "Hello world"
    assert info["model"] == "m2"


def test_stream_skips_a_silent_model():
    client = FakeClient({"m1": [[""]], "m2": [["text"]]})
    assert "".join(make_rag(client=client).stream(MESSAGES)) == "text"


def test_stream_says_busy_when_every_model_fails():
    client = FakeClient({m: [status_error(503)] * 2 for m in ("m1", "m2")})
    assert "".join(make_rag(client=client).stream(MESSAGES)) == config.BUSY_MESSAGE


# ---------- practice questions ----------
SLIDES = CHUNKS[:3]


@pytest.fixture
def short_slides_ok(monkeypatch):
    monkeypatch.setattr(config, "MIN_CHARS", 10)  # the test slides are short


def mcq(**kw):
    q = {"type": "mcq", "question": "What is a hallucination?", "cite": "L01 slide 1",
         "options": ["Unsupported fluent output", "A tokenizer", "A cosine", "An index"],
         "answer": 0, "explanation": "Slide 1 defines it."}
    return {**q, **kw}


def test_clean_quiz_keeps_valid_questions_and_shuffles_safely():
    for _ in range(10):
        [q] = clean_quiz(json.dumps({"questions": [mcq()]}), SLIDES)
        assert q["options"][q["answer"]] == "Unsupported fluent output"
        assert q["hit"]["cite"] == "L01 slide 1"


def test_clean_quiz_rejects_bad_questions():
    bad = [mcq(answer=7), mcq(options=["same", "same", "x", "y"]), mcq(cite="L09 slide 99"),
           mcq(type="essay"), {"type": "short", "question": "Why?", "answer": "",
                                "cite": "L01 slide 2"}]
    assert clean_quiz(json.dumps({"questions": bad}), SLIDES) == []


def test_clean_quiz_handles_code_fences_duplicates_and_garbage():
    text = "```json\n" + json.dumps({"questions": [mcq(), mcq()]}) + "\n```"
    assert len(clean_quiz(text, SLIDES)) == 1
    assert clean_quiz("not json at all", SLIDES) == []
    assert clean_quiz(json.dumps({"questions": [mcq()]}), SLIDES, kind="short") == []


def test_find_slide_inside_merged_chunk():
    merged = {**CHUNKS[0], "pages": [13, 14], "labels": [13, 14], "cite": "L01 slide 13-14"}
    by_cite = {merged["cite"]: merged}
    assert find_slide("[L01 slide 14]", by_cite, [merged]) is merged
    assert find_slide("L01 p.14", by_cite, [merged]) is merged
    assert find_slide("L01 slide 15", by_cite, [merged]) is None


def test_generate_quiz_end_to_end(short_slides_ok):
    reply = json.dumps({"questions": [mcq(), {"type": "short", "question": "Define temperature.",
                                               "answer": "Sampling randomness.", "cite": "L01 slide 2",
                                               "explanation": "Mention randomness."}]})
    info = {}
    qs = make_rag(client=FakeClient(default=reply)).generate_quiz(
        2, lectures=["L01"], info=info)
    assert [q["type"] for q in qs] == ["mcq", "short"] and info["model"] == "m1"


def test_generate_quiz_busy_returns_none(short_slides_ok):
    client = FakeClient({m: [status_error(503)] * 2 for m in ("m1", "m2")})
    assert make_rag(client=client).generate_quiz(2) is None
