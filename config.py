"""Shared settings. Change values here instead of inside the scripts."""
import os
from dotenv import load_dotenv

load_dotenv()

SLIDES_DIR = "data/slides"        # put lecture PDFs here, e.g. L03_RAG_basics.pdf
INDEX_DIR = "index"               # ingest.py writes chunks.json + embeddings.npy here

# Free local embedding model that handles English and Korean.
EMBED_MODEL = "intfloat/multilingual-e5-small"

# ---------- AI providers ----------
# Any OpenAI-compatible service works. A provider is used only when its API key is set (.env
# locally, Secrets on Streamlit Cloud). A model is written "provider:model", e.g.
# "groq:qwen/qwen3.8-27b"; a name without a provider ("gemini-3.5-flash-lite") belongs to the
# main provider (Gemini, or OpenAI if only OPENAI_API_KEY is set).
#   auto:  models added to the "Auto" fallback chain, after the main provider's models
#          (override with e.g. GROQ_MODELS=a,b in .env)
#   show:  which of the provider's models appear in the model picker (regex on the model id);
#          the list itself is fetched from the provider, so it stays current
# Free tiers (Nov 2026): Groq 1,000 requests/day per model (qwen3.8-27b, gpt-oss-120b/20b);
# OpenRouter 50 requests/day in total on its ":free" models (1,000 after buying $10 of credit).
PROVIDERS = {
    "gemini": {"label": "Gemini", "key_env": "GEMINI_API_KEY",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    "groq": {"label": "Groq", "key_env": "GROQ_API_KEY",
             "base_url": "https://api.groq.com/openai/v1",
             "auto": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
             "show": r"^(?!.*(whisper|orpheus|guard|tts|playai|distil|compound)).*$"},
    "openrouter": {"label": "OpenRouter", "key_env": "OPENROUTER_API_KEY",
                   "base_url": "https://openrouter.ai/api/v1",
                   "auto": ["google/gemma-4-31b-it:free"],
                   "show": r":free$"},
    "openai": {"label": "OpenAI", "key_env": "OPENAI_API_KEY", "base_url": None},
}
for _name, _p in PROVIDERS.items():
    _p["key"] = os.getenv(_p["key_env"])
    _env_models = os.getenv(f"{_name.upper()}_MODELS")
    if _env_models:
        _p["auto"] = [m.strip() for m in _env_models.split(",") if m.strip()]

MAIN_PROVIDER = "gemini" if PROVIDERS["gemini"]["key"] or not PROVIDERS["openai"]["key"] \
    else "openai"
LLM_API_KEY = PROVIDERS[MAIN_PROVIDER]["key"]
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or PROVIDERS[MAIN_PROVIDER]["base_url"]
PROVIDERS[MAIN_PROVIDER]["base_url"] = LLM_BASE_URL

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
# Tried in order if the main model is busy (503), rate-limited (429), unavailable (404) or too
# slow (LLM_TIMEOUT). The free tier has a separate DAILY request quota per model (e.g. 500 for
# the flash-lite models, 20 for gemini-3.5-flash), so listing several models also multiplies the
# daily budget. Duplicates of LLM_MODEL are skipped. In testing (Nov 2026) gemini-3.5-flash was
# often overloaded, and gemini-flash-latest / 3.7 / 3.8-flash timed out.
FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "FALLBACK_MODELS",
    "gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-3.1-flash-lite,gemini-3.5-flash"
).split(",") if m.strip()]


def available_providers() -> list:
    """Providers with an API key, main provider first."""
    return [MAIN_PROVIDER] + [n for n, p in PROVIDERS.items() if p["key"] and n != MAIN_PROVIDER]


# Tried first by Auto when their provider has a key. gpt-oss-120b on Groq scored as well as
# Gemini on the test set (100% vs 97.6% correct) and answers in 1.0 s instead of 2.7 s, but its
# free tier allows only ~200,000 tokens (~80 answers) a day; after that Auto skips it until
# Groq's "try again in ..." time and Gemini answers.
PREFERRED_MODELS = [m.strip() for m in os.getenv(
    "PREFERRED_MODELS", "groq:openai/gpt-oss-120b").split(",") if m.strip()]

LLM_TIMEOUT = 20          # seconds to wait for a model before trying the next one
COOLDOWN_SECONDS = 120    # skip a model that just failed for this long (shared by all visitors)

# Website uploads: indexing runs on the server's CPU, so cap the work per visitor.
MAX_UPLOAD_PAGES = 300   # total pages across all PDFs one visitor uploads
                         # (per-file size limit: maxUploadSize in .streamlit/config.toml)

MIN_CHARS = 80    # slides shorter than this get merged with the next slide
TOP_K = 8          # how many slides to retrieve per question. 8 rather than 5 since all 9
                   # lectures (491 slides) are loaded: on the 47-question test set it removed
                   # false refusals (5% -> 0%) and raised correct answers (90% -> 100%).
# Combine meaning-based search with keyword search (BM25). Helps questions whose key word
# appears on the slide but whose meaning is vague, e.g. "What must students submit?".
HYBRID_SEARCH = True
# Count a slide's title (read from the PDF) twice in keyword search? Tested on the 40 answerable
# test questions (k=8): the right slide in the top 8 for 39 -> 38, MRR 0.840 -> 0.821 (better
# top 3: 36 -> 38). Not better overall, so off. `evaluate.py --retrieval-only` measures it.
TITLE_BOOST = False
# Refuse without calling the LLM if the best score is below this. Measured on eval/test_set.jsonl:
# in-scope 0.788-0.906, out-of-scope 0.752-0.837. They overlap, so 0.78 only blocks clearly
# off-topic questions; the prompt's refusal rule handles the rest.
MIN_SCORE = 0.78

# When the question and the slides are in different languages (e.g. a Korean question about
# English slides), translate the question before searching. Cross-language search alone ranked
# the wrong slides and scored Korean questions below MIN_SCORE, so they were refused.
TRANSLATE_QUERIES = True

REFUSAL = "This isn't covered in the lecture slides I have."
REFUSAL_KO = "강의 슬라이드에서 다루지 않는 내용입니다."
REFUSAL_RU = "Этого нет в слайдах лекций, которые у меня есть."
REFUSALS = {"en": REFUSAL, "ko": REFUSAL_KO, "ru": REFUSAL_RU}  # by the question's language
BUSY_MESSAGE = "The AI service is busy right now. Please try again in a minute."

# Practice questions ("Practice exam" tab)
QUIZ_MAX_SLIDES = 20     # slides sent to the model per quiz
QUIZ_TIMEOUT = 30        # writing 15 questions takes ~5-10 s on flash-lite; longer than one answer

# One-click starter questions on the website (label shown on the chip -> question asked).
# warm_cache.py prepares answers for these so the demo is instant.
DEMO_QUESTIONS = {
    ":material/help: What is a hallucination?": "What is a hallucination?",
    ":material/thermostat: Does temperature make answers more accurate?":
        "Does raising the temperature make the model's answers more accurate?",
    ":material/analytics: How is recall@k calculated?": "How is recall@k calculated?",
    ":material/account_tree: Workflow vs agent":
        "What is the difference between a fixed workflow and an agent?",
    ":material/translate: 환각이란 무엇인가요?": "환각이란 무엇인가요?",
}
# The same starter questions when the interface is in Russian (also prepared by warm_cache.py).
DEMO_QUESTIONS_RU = {
    ":material/help: Что такое галлюцинация?": "Что такое галлюцинация?",
    ":material/thermostat: Делает ли температура ответы точнее?":
        "Делает ли повышение температуры ответы модели точнее?",
    ":material/analytics: Как считается recall@k?": "Как вычисляется recall@k?",
    ":material/account_tree: Workflow или агент?":
        "Чем фиксированный workflow отличается от агента?",
    ":material/translate: What is a hallucination?": "What is a hallucination?",
}
