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
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
# Tried in order if the main model is busy (503), rate-limited (429) or unavailable (404).
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "gemini-3.1-flash-lite").split(",") if m.strip()]

# Website uploads: indexing runs on the server's CPU, so cap the work per visitor.
MAX_UPLOAD_PAGES = 300   # total pages across all PDFs one visitor uploads
                         # (per-file size limit: maxUploadSize in .streamlit/config.toml)

MIN_CHARS = 80    # slides shorter than this get merged with the next slide
TOP_K = 5          # how many slides to retrieve per question
# Refuse without calling the LLM if the best score is below this. Measured on eval/test_set.jsonl:
# in-scope 0.788-0.906, out-of-scope 0.752-0.837. They overlap, so 0.78 only blocks clearly
# off-topic questions; the prompt's refusal rule handles the rest.
MIN_SCORE = 0.78

REFUSAL = "This isn't covered in the lecture slides I have."
BUSY_MESSAGE = "The AI service is busy right now. Please try again in a minute."
