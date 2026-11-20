"""Saved answers, so repeated questions are instant and still work when the AI service is busy.

Two layers, both only for a chat's first question (follow-ups depend on the conversation):
  - index/answer_cache.json: answers prepared in advance by `python warm_cache.py` (the starter
    questions and the demo path). Used only while index/chunks.json is unchanged.
  - an in-memory cache on the server (app.py): any first question someone has asked.
An entry stores the answer, the slides it searched (by cite tag) and how it was found.

index/summaries.json works the same way for the lecture summaries (Summary tab), one per
lecture and interface language.
"""
import hashlib
import json
from pathlib import Path

import config

CACHE_FILE = Path(config.INDEX_DIR) / "answer_cache.json"
SUMMARY_FILE = Path(config.INDEX_DIR) / "summaries.json"


def index_fingerprint() -> str:
    """Changes whenever the demo index is rebuilt, so old answers aren't reused. Line endings
    are ignored: Git on Windows checks the file out with \\r\\n, the server gets \\n."""
    try:
        data = (Path(config.INDEX_DIR) / "chunks.json").read_bytes().replace(b"\r\n", b"\n")
    except OSError:
        return ""
    return hashlib.sha1(data).hexdigest()[:16]


def cache_key(question: str, k: int, lectures=None) -> str:
    q = " ".join(question.split()).lower()
    return f"{q}|k={k}|lectures={','.join(sorted(lectures or []))}"


def to_entry(answer: str, hits: list, trace: dict) -> dict:
    return {"answer": answer, "trace": trace,
            "hits": [{"cite": h["cite"], "pdf": h["pdf"], "score": round(h["score"], 4)}
                     for h in hits]}


def from_entry(entry: dict, chunks: list):
    """(answer, hits, trace), or None if a slide it refers to no longer exists."""
    by_id = {(c["cite"], c["pdf"]): c for c in chunks}
    hits = []
    for h in entry["hits"]:
        chunk = by_id.get((h["cite"], h["pdf"]))
        if chunk is None:
            return None
        hits.append({**chunk, "score": h["score"]})
    return entry["answer"], hits, {**entry["trace"], "cached": True}


def _load(path: Path, field: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get(field, {}) if data.get("index") == index_fingerprint() else {}


def _save(path: Path, field: str, entries: dict):
    path.write_text(json.dumps({"index": index_fingerprint(), field: entries},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def load_file() -> dict:
    """Saved answers for the current demo index ({} if missing or out of date)."""
    return _load(CACHE_FILE, "answers")


def save_file(answers: dict):
    _save(CACHE_FILE, "answers", answers)


def summary_key(tag: str, lang: str) -> str:
    return f"{tag}|{lang}"


def load_summaries() -> dict:
    """Saved lecture summaries for the current demo index: {summary_key: entry}, where an entry
    is {"text", "model", "seconds", "slides", "trimmed"}."""
    return _load(SUMMARY_FILE, "summaries")


def save_summaries(summaries: dict):
    _save(SUMMARY_FILE, "summaries", summaries)
