"""Shared settings. Change values here instead of inside the scripts."""
import os
from dotenv import load_dotenv

load_dotenv()

SLIDES_DIR = "data/slides"        # put lecture PDFs here, e.g. L03_RAG_basics.pdf
INDEX_DIR = "index"               # ingest.py writes chunks.json + embeddings.npy here

# Free local embedding model that handles English and Korean.
EMBED_MODEL = "intfloat/multilingual-e5-small"

# LLM used for answering. Works with any OpenAI-compatible API.
# Gemini: set GEMINI_API_KEY + LLM_BASE_URL in .env.  OpenAI: set OPENAI_API_KEY only.
LLM_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
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
