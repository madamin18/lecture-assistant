"""Several AI providers: model references, the Auto chain, choosing a model, provider failures,
and hiding <think> reasoning."""
import time
from types import SimpleNamespace

import openai
import pytest

import config
import rag
from conftest import FakeClient, make_rag, status_error
from rag import (auto_chain, canonical, list_models, model_label, split_ref, strip_thinking,
                 without_thinking)

MESSAGES = [{"role": "user", "content": "hi"}]
QWEN = "groq:qwen/qwen3.8-27b"
OSS = "groq:openai/gpt-oss-120b"


@pytest.fixture
def groq(monkeypatch):
    """Groq has a key; its Auto models are gpt-oss-120b then qwen."""
    monkeypatch.setitem(config.PROVIDERS["groq"], "key", "test-key")
    monkeypatch.setitem(config.PROVIDERS["groq"], "auto", ["openai/gpt-oss-120b",
                                                           "qwen/qwen3.8-27b"])


def two_providers(gemini_script=None, groq_script=None, groq_default="from groq"):
    gemini = FakeClient(gemini_script, default="from gemini")
    grq = FakeClient(groq_script, default=groq_default)
    return make_rag(client={"gemini": gemini, "groq": grq}), gemini, grq


def test_model_references():
    assert split_ref(QWEN) == ("groq", "qwen/qwen3.8-27b")
    assert split_ref("m1") == ("gemini", "m1")
    # an OpenRouter id has its own colon, but no provider prefix -> main provider
    assert split_ref("google/gemma-4-31b-it:free") == ("gemini", "google/gemma-4-31b-it:free")
    assert canonical("gemini:m1") == "m1" and canonical(QWEN) == QWEN
    assert model_label(QWEN) == "Groq · qwen/qwen3.8-27b"


def test_auto_chain_adds_providers_with_a_key(groq):
    assert auto_chain() == ["m1", "m2", OSS, QWEN]


def test_auto_chain_without_other_keys():
    assert auto_chain() == ["m1", "m2"]


def test_main_provider_out_of_quota_falls_back_to_groq(groq):
    daily = status_error(429, '{"retryDelay": "40000s"}')
    rag_, gemini, grq = two_providers({"m1": [daily], "m2": [daily]})
    info = {}
    assert rag_.chat(MESSAGES, info) == "from groq"
    assert info["model"] == OSS and grq.calls[0]["model"] == "openai/gpt-oss-120b"


def test_chosen_model_goes_first(groq):
    rag_, gemini, grq = two_providers()
    info = {}
    assert rag_.chat(MESSAGES, info, model=QWEN) == "from groq"
    assert info["model"] == QWEN and gemini.calls == []


def test_chosen_model_busy_falls_back_unless_told_not_to(groq):
    rag_, gemini, grq = two_providers(groq_script={"qwen/qwen3.8-27b": [status_error(503)]})
    info = {}
    assert rag_.chat(MESSAGES, info, model=QWEN) == "from gemini" and info["model"] == "m1"
    assert info["failed"] == [[QWEN, "overloaded"]]  # shown as a step in "How I found this"

    rag._cooldown_until.clear()
    rag_, gemini, grq = two_providers(groq_script={"qwen/qwen3.8-27b": [status_error(503)] * 2})
    info = {}
    assert rag_.chat(MESSAGES, info, model=QWEN, fallback=False) is None  # compare mode
    assert info["error"] == "overloaded" and gemini.calls == []


def test_bad_key_on_one_provider_does_not_stop_the_others(groq):
    rag_, gemini, grq = two_providers({"m1": [status_error(401)]})
    info = {}
    assert rag_.chat(MESSAGES, info) == "from groq"
    assert [c["model"] for c in gemini.calls] == ["m1"]  # m2 skipped: same provider, same key
    assert rag._cooldown_until["gemini:*"] > time.time()


def test_bad_key_everywhere_is_raised():
    client = FakeClient({"m1": [status_error(401)]})
    with pytest.raises(openai.APIStatusError):
        make_rag(client=client).chat(MESSAGES)


def test_rejected_request_tries_next_model_without_cooldown():
    client = FakeClient({"m1": [status_error(413)]}, default="m2 ok")
    info = {}
    assert make_rag(client=client).chat(MESSAGES, info) == "m2 ok" and info["model"] == "m2"
    assert "m1" not in rag._cooldown_until


def test_unknown_model_is_skipped_for_an_hour():
    client = FakeClient({"m1": [status_error(404)]})
    make_rag(client=client).chat(MESSAGES)
    assert rag._cooldown_until["m1"] - time.time() > 3500


def test_list_models_filters_and_falls_back(groq):
    ids = ["whisper-large-v3", "qwen/qwen3.8-27b", "openai/gpt-oss-120b",
           "meta-llama/llama-prompt-guard-2-22m"]
    client = SimpleNamespace(models=SimpleNamespace(
        list=lambda: [SimpleNamespace(id=i) for i in ids]))
    assert list_models("groq", client) == [OSS, QWEN]

    def broken():
        raise RuntimeError("no network")
    client.models.list = broken
    assert list_models("groq", client) == [OSS, QWEN]  # the configured Auto models
    assert list_models("gemini", None) == ["m1", "m2"]


def test_thinking_is_hidden():
    assert strip_thinking("<think>Let me see...</think>\n\nRAG is [L01 slide 3].") == \
        "RAG is [L01 slide 3]."
    assert strip_thinking("<think>never closed") == ""
    pieces = ["<thi", "nk>hmm, the slide", " says</th", "ink>\nAnswer ", "with <b>bold</b>."]
    assert "".join(without_thinking(pieces)) == "\nAnswer with <b>bold</b>."
    assert "".join(without_thinking(["no tags here", " <", "3"])) == "no tags here <3"


def test_stream_hides_thinking_and_reports_model():
    client = FakeClient({"m1": [["<think>reasoning", "...</think>", "\n\nThe answer."]]})
    info = {}
    assert "".join(make_rag(client=client).stream(MESSAGES, info)) == "The answer."
    assert info["model"] == "m1"
