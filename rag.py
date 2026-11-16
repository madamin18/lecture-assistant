"""Step 2: retrieval + answer generation with slide citations."""
import json
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz
from openai import OpenAI, APIStatusError, APIConnectionError
from sentence_transformers import SentenceTransformer

import config
from ingest import open_pdf

SYSTEM_PROMPT = f"""You are a study assistant for a university course.
Answer the student's question using ONLY the lecture slide excerpts provided.

Rules:
- After every sentence that uses a slide, cite it with its tag in square brackets,
  exactly as given, e.g. [L03 p.14]. One tag per bracket.
- If the excerpts do not contain the answer, reply with exactly:
  "{config.REFUSAL}"
- Never use outside knowledge, and never invent slide numbers.
- Be concise. Answer in the same language as the question."""


def make_client() -> OpenAI:
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                  max_retries=1, timeout=20)  # fail fast so the fallback models get a turn


class SlideRAG:
    """Search + answer over one set of slides.

    SlideRAG()                         loads the demo index from index/
    SlideRAG(chunks=..., emb=...,      uses an index built in memory (uploaded PDFs);
             pdf_data={name: bytes})   pdf_data lets the UI render those slides
    Pass embedder/client to share one copy between many SlideRAG objects (the web app does).
    """

    def __init__(self, index_dir: str = config.INDEX_DIR, chunks=None, emb=None,
                 pdf_data=None, embedder=None, client=None):
        if chunks is None:
            d = Path(index_dir)
            chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
            emb = np.load(d / "embeddings.npy")
        self.chunks, self.emb = chunks, emb
        self.pdf_data = pdf_data or {}  # uploaded file name -> PDF bytes
        self.model = embedder or SentenceTransformer(config.EMBED_MODEL)
        self.client = client or make_client()
        self.last_model = None  # which model produced the last answer

    def retrieve(self, question: str, k: int = config.TOP_K) -> list:
        if not self.chunks:
            return []
        q = self.model.encode([f"query: {question}"], normalize_embeddings=True)[0]
        scores = self.emb @ q  # cosine similarity (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def answer(self, question: str, k: int = config.TOP_K) -> tuple[str, list]:
        self.last_model = None
        hits = self.retrieve(question, k)
        if not hits or hits[0]["score"] < config.MIN_SCORE:
            return config.REFUSAL, hits

        context = "\n\n".join(f"[{h['cite']}] ({h['title']})\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Slide excerpts:\n{context}\n\nQuestion: {question}"},
        ]
        reply = self.chat(messages)
        return (reply if reply is not None else config.BUSY_MESSAGE), hits

    def chat(self, messages: list):
        """Call the main model, falling back to the others. Returns None if all are busy."""
        for model in [config.LLM_MODEL] + config.FALLBACK_MODELS:
            try:
                resp = self.client.chat.completions.create(
                    model=model, temperature=0, messages=messages)
                self.last_model = model
                return resp.choices[0].message.content.strip()
            except (APIStatusError, APIConnectionError) as e:
                status = getattr(e, "status_code", "connection")
                print(f"[warn] {model} failed ({status}), trying next model...")
                if isinstance(e, APIStatusError) and e.status_code in (400, 401, 403):
                    raise  # bad request or bad key: retrying other models won't help
        return None


def cited_hits(answer: str, hits: list) -> list:
    """The retrieved slides the answer actually cites, in the order they were retrieved."""
    # (?![\d-]) stops "L01 p.1" from matching inside "L01 p.13" or "L01 p.1-2"
    return [h for h in hits if re.search(re.escape(h["cite"]) + r"(?![\d-])", answer)]


def render_page(pdf_ref: str, page_no: int, data: bytes | None = None, zoom: float = 1.2) -> bytes:
    """PNG image of one slide. pdf_ref is a path on disk, or an upload's name when data is given."""
    if data is None:
        pdf_ref = pdf_ref.replace("\\", "/")  # indexes built on Windows still load on Linux
    with open_pdf(pdf_ref, data) as doc:
        pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")


if __name__ == "__main__":  # quick terminal test: python rag.py "What is RAG?"
    import sys
    rag = SlideRAG()
    ans, hits = rag.answer(" ".join(sys.argv[1:]) or "What is retrieval-augmented generation?")
    print(f"(model: {rag.last_model})\n")
    print(ans, "\n\nRetrieved:")
    for h in hits:
        print(f"  [{h['cite']}] score={h['score']:.3f}")
