"""Lecture summaries (Summary tab): what is sent to the AI, which model writes it, and the saved
summaries file."""
import pytest

import answer_cache
import config
import rag
from conftest import CHUNKS, FakeClient, chunk, make_rag, status_error
from ui import i18n


def test_fit_cap_keeps_short_lectures_whole():
    assert rag.fit_cap([100, 300, 50], 1000) == 300
    assert rag.fit_cap([], 1000) == 0


def test_fit_cap_cuts_only_the_longest_slides():
    lengths = [100, 2000, 3000, 50]
    cap = rag.fit_cap(lengths, 2000)
    assert sum(min(n, cap) for n in lengths) <= 2000
    assert sum(min(n, cap + 1) for n in lengths) > 2000  # as long as the budget allows
    assert cap > 100  # the short slides stay whole


@pytest.mark.parametrize("lang", list(i18n.LANGUAGES))
def test_summary_has_headings_and_term_rule_in_every_interface_language(lang):
    messages, slides = make_rag().summary_messages("L01", lang)
    prompt = messages[0]["content"]
    assert rag.LANGUAGE_NAMES[lang] in prompt
    assert all(f"### {h}" in prompt for h in config.SUMMARY_HEADINGS[lang])
    assert rag.SUMMARY_TERMS[lang] in prompt


def test_summary_sends_only_that_lecture_in_order():
    info = {}
    messages, slides = make_rag().summary_messages("L01", "en", info)
    user = messages[1]["content"]
    assert [s["cite"] for s in slides] == ["L01 slide 1", "L01 slide 2", "L01 slide 3"]
    assert user.index("[L01 slide 1]") < user.index("[L01 slide 2]") < user.index("[L01 slide 3]")
    assert "L02" not in user and "Tokenizers" not in user
    assert user.startswith("Lecture L01: L01 title")
    assert info == {"slides": 3, "trimmed": False}


def test_long_lecture_is_shortened_to_the_budget(monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_MAX_CHARS", 600)
    chunks = [chunk("L05", p, f"slide {p} " + "word " * (20 if p == 1 else 200))
              for p in range(1, 4)]
    info = {}
    messages, _ = make_rag(chunks).summary_messages("L05", "en", info)
    assert info == {"slides": 3, "trimmed": True}
    assert chunks[0]["text"] in messages[1]["content"]  # the short slide is kept whole
    assert len(messages[1]["content"]) < 600 + 100  # slide text within budget, plus headers


def test_summary_starts_with_the_summary_model_then_falls_back(monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_MODEL", "m2")
    client = FakeClient({"m2": [status_error(503)], "m1": [["### Overview", " text"]]})
    info = {}
    text = "".join(make_rag(client=client).summarize("L01", "en", info))
    assert text == "### Overview text"
    assert [c["model"] for c in client.calls] == ["m2", "m1"]
    assert info["model"] == "m1" and info["slides"] == 3


def test_summary_uses_the_model_picked_in_the_sidebar(monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_MODEL", "m1")
    client = FakeClient(default="summary")
    "".join(make_rag(client=client).summarize("L02", "ru", model="m2"))
    assert client.calls[0]["model"] == "m2"


def test_summary_says_busy_when_no_model_answers():
    client = FakeClient({m: [status_error(503)] * 2 for m in ("m1", "m2")})
    info = {}
    text = "".join(make_rag(client=client).summarize("L01", "en", info))
    assert text == config.BUSY_MESSAGE and info.get("model") is None


def test_saved_summaries_only_used_for_the_same_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(answer_cache, "SUMMARY_FILE", tmp_path / "summaries.json")
    (tmp_path / "chunks.json").write_text('[{"a": 1}]')
    key = answer_cache.summary_key("L01", "ru")
    answer_cache.save_summaries({key: {"text": "Обзор", "model": "m1"}})
    assert answer_cache.load_summaries() == {key: {"text": "Обзор", "model": "m1"}}
    (tmp_path / "chunks.json").write_text('[{"a": 2}]')  # rebuilt index
    assert answer_cache.load_summaries() == {}


def test_summary_citations_are_checked_against_the_lecture():
    slides = [c for c in CHUNKS if c["lecture"] == "L01"]
    text = "- Point [L01 slide 1]\n- Other [L01 slide 3; L02 slide 1]"
    facts = rag.citation_facts(text, slides)
    assert facts["cited"] == 2 and facts["made_up"] == 1  # L02 isn't part of this lecture
