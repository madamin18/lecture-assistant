"""Shared test helpers. The tests need no API key, no internet and no embedding model: a fake
embedder (word hashing) and a fake AI client stand in for the real ones."""
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openai
import pytest

try:
    import httpx2 as httpx  # the HTTP library of openai >= 3
except ImportError:
    import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import rag  # noqa: E402


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words get similar vectors."""
    dim = 256

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in rag.tokenize(text.split(": ", 1)[-1]):  # drop "query: " / "passage: "
                out[row, zlib.crc32(word.encode()) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def get_sentence_embedding_dimension(self):
        return self.dim


def reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def stream_of(*pieces):
    return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))])
                 for p in pieces])


def status_error(code, text=""):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.APIStatusError(f"error {code}", body=None,
                                 response=httpx.Response(code, request=request, text=text))


def timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))


class FakeClient:
    """client.chat.completions.create(...) like the OpenAI SDK. `script` maps a model name to the
    results it gives, in order: a string (the reply), a list of strings (a streamed reply), or
    an exception to raise. Every call is recorded in `calls`."""

    def __init__(self, script=None, default="OK"):
        self.script = {m: list(r) for m, r in (script or {}).items()}
        self.default = default
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, stream=False, **kwargs):
        self.calls.append({"model": model, "messages": messages, "stream": stream, **kwargs})
        queue = self.script.get(model)
        result = queue.pop(0) if queue else self.default
        if isinstance(result, Exception):
            raise result
        if stream:
            return stream_of(*([result] if isinstance(result, str) else result))
        return reply(result)


def chunk(lecture, page, text, heading="", label=None):
    label = page if label is None else label
    return {"lecture": lecture, "title": f"{lecture} title", "pdf": f"{lecture}.pdf",
            "pages": [page], "labels": [label], "heading": heading, "headings": [heading],
            "text": text, "cite": f"{lecture} slide {label}"}


CHUNKS = [
    chunk("L01", 1, "A hallucination is fluent output that is not supported by evidence."),
    chunk("L01", 2, "Temperature changes randomness of sampling, not factual accuracy."),
    chunk("L01", 3, "Students must submit the lab evidence folder before the deadline."),
    chunk("L02", 1, "Tokenizers split text into tokens; token boundaries depend on the tokenizer."),
    chunk("L02", 2, "Cosine similarity measures the angle between embedding vectors."),
    chunk("L03", 1, "Chunk overlap repeats some text between neighbouring chunks."),
]


def make_rag(chunks=CHUNKS, client=None):
    embedder = FakeEmbedder()
    emb = embedder.encode([f"passage: {c['text']}" for c in chunks])
    return rag.SlideRAG(chunks=chunks, emb=emb, embedder=embedder, client=client or FakeClient())


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Two fake models from the main provider only (keys in .env don't matter), no cooldowns
    left over from other tests, and no real waiting."""
    monkeypatch.setattr(config, "LLM_MODEL", "m1")
    monkeypatch.setattr(config, "FALLBACK_MODELS", ["m1", "m2"])
    monkeypatch.setattr(config, "MAIN_PROVIDER", "gemini")
    for provider in config.PROVIDERS.values():
        monkeypatch.setitem(provider, "key", None)
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)
    rag._cooldown_until.clear()
    yield
    rag._cooldown_until.clear()
