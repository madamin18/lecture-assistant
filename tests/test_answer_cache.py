"""Saved answers: keys, round trip, and invalidation when the index changes."""
import json

import answer_cache
import config
from conftest import CHUNKS


def test_cache_key_ignores_case_spacing_and_filter_order():
    a = answer_cache.cache_key("What  is RAG?", 8, ["L02", "L01"])
    assert a == answer_cache.cache_key("what is rag?", 8, ["L01", "L02"])
    assert a != answer_cache.cache_key("what is rag?", 5, ["L01", "L02"])


def test_entry_round_trip_and_missing_slide():
    hits = [{**CHUNKS[0], "score": 0.91234}, {**CHUNKS[3], "score": 0.8}]
    entry = answer_cache.to_entry("Answer [L01 slide 1]", hits, {"k": 8, "model": "m1"})
    answer, back, trace = answer_cache.from_entry(json.loads(json.dumps(entry)), CHUNKS)
    assert answer == "Answer [L01 slide 1]" and trace["cached"] is True
    assert [h["cite"] for h in back] == ["L01 slide 1", "L02 slide 1"]
    assert back[0]["text"] == CHUNKS[0]["text"] and back[0]["score"] == 0.9123
    assert answer_cache.from_entry(entry, CHUNKS[1:]) is None  # that slide is gone


def test_saved_answers_only_used_for_the_same_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(answer_cache, "CACHE_FILE", tmp_path / "answer_cache.json")
    (tmp_path / "chunks.json").write_bytes(b'[{"a": 1},\n{"b": 2}]')
    answer_cache.save_file({"q|k=8|lectures=": {"answer": "x"}})
    assert answer_cache.load_file() == {"q|k=8|lectures=": {"answer": "x"}}

    # Git on Windows checks the index out with \r\n: still the same index
    (tmp_path / "chunks.json").write_bytes(b'[{"a": 1},\r\n{"b": 2}]')
    assert answer_cache.load_file() != {}

    (tmp_path / "chunks.json").write_bytes(b'[{"a": 2}]')  # rebuilt index
    assert answer_cache.load_file() == {}
