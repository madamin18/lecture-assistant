"""Step 2: retrieval + answer generation with slide citations, plus practice-question generation."""
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
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
- Earlier messages in the chat are only there so you know what a follow-up question
  refers to ("it", "that", "explain more"). The facts must still come from the excerpts.
- Be concise. Answer in the same language as the question."""

QUIZ_PROMPT = """You write exam practice questions for university students, using ONLY the
lecture slide excerpts provided. Every question must be answerable from the slide it cites.

Write {n} questions. {mix}
Return JSON only, in exactly this shape:
{{"questions": [
  {{"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."], "answer": 0,
    "explanation": "why the answer is right, in 1-2 sentences", "cite": "L01 p.14"}},
  {{"type": "short", "question": "...", "answer": "a model answer in 1-3 sentences",
    "explanation": "what a full-mark answer must mention", "cite": "L01 p.14"}}
]}}

Rules:
- "mcq" has exactly 4 options and exactly one correct option; "answer" is its index (0-3).
  Wrong options must be plausible to a student who skimmed the slides. Avoid "all of the
  above" / "none of the above" (the options get shuffled).
- "cite" is the tag of the one slide the answer comes from, copied exactly from the excerpts.
- Test understanding: definitions, but also why, compare, and apply-to-an-example questions.
- Cover different slides; never ask two questions about the same fact.
- Skip slide footers, page numbers, layout, and title or agenda slides.
- Write the questions in the same language as the slides."""

QUIZ_MIX = {
    "mixed": "Make about two thirds multiple choice (\"mcq\") and the rest short answer (\"short\").",
    "mcq": "Make them all multiple choice (\"mcq\").",
    "short": "Make them all short answer (\"short\").",
}

# Common words that would otherwise count as keyword matches.
STOPWORDS = set("""a an and are as at be by can do does did for from has have how i in is it its
of on or that the their them they this to was what when where which who why will with would
you your about into than then there these those should could must what's s""".split())

# Models that recently failed (timeout, 503, 429) are skipped for a while, so one overloaded
# model doesn't make every visitor wait for its timeout. Shared by all visitors.
_cooldown_until: dict[str, float] = {}


def tokenize(text: str) -> list:
    """Lowercase keyword tokens. Non-Latin words (e.g. Korean) also get 2-character pieces,
    so '프롬프트의' still matches '프롬프트'."""
    out = []
    for tok in re.findall(r"\w+", text.lower()):
        if tok in STOPWORDS:
            continue
        if tok.isascii():
            if len(tok) > 1:
                out.append(tok)
        else:
            out.append(tok)
            out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return out


class BM25:
    """Keyword search (Okapi BM25). Finds slides that share rare exact words with the question,
    which meaning-based search can miss (e.g. 'submit' -> the 'Submit evidence' slide)."""

    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.lengths.mean()) if docs else 1.0
        self.postings = defaultdict(list)  # word -> [(doc index, count)]
        for i, d in enumerate(docs):
            for word, count in Counter(d).items():
                self.postings[word].append((i, count))

    def scores(self, query: list) -> np.ndarray:
        s = np.zeros(self.n, dtype="float32")
        for word in set(query):
            posts = self.postings.get(word)
            if not posts:
                continue
            idf = math.log(1 + (self.n - len(posts) + 0.5) / (len(posts) + 0.5))
            for i, f in posts:
                s[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
        return s


def rrf_ranks(scores: np.ndarray, only_positive: bool = False) -> np.ndarray:
    """Reciprocal-rank-fusion contribution 1/(60 + rank) for each slide."""
    order = np.argsort(-scores)
    contrib = np.zeros(len(scores), dtype="float32")
    for rank, i in enumerate(order, start=1):
        if only_positive and scores[i] <= 0:
            break
        contrib[i] = 1.0 / (60 + rank)
    return contrib


def make_client() -> OpenAI:
    # No automatic retries: on failure we move straight to the next model in the list.
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                  max_retries=0, timeout=config.LLM_TIMEOUT)


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
        self.bm25 = BM25([tokenize(f"{c['title']} {c['text']}") for c in chunks])
        self.pdf_data = pdf_data or {}  # uploaded file name -> PDF bytes
        self.model = embedder or SentenceTransformer(config.EMBED_MODEL)
        self.client = client or make_client()
        self.last_model = None  # which model produced the last answer

    @property
    def lectures(self) -> list:
        """[(lecture tag, title)] in index order, e.g. [('L01', 'Intro')]."""
        seen = {}
        for c in self.chunks:
            seen.setdefault(c["lecture"], c["title"])
        return list(seen.items())

    # ---------- search ----------
    def retrieve(self, question: str, k: int = config.TOP_K, previous: str | None = None) -> list:
        """Top-k slides. `previous` = the student's previous question, used so follow-ups like
        "explain that more simply" still find the right slides.
        Each hit's "score" is the meaning-based (cosine) similarity, used for the refusal check."""
        if not self.chunks:
            return []
        queries = [question] + ([f"{previous} {question}"] if previous else [])
        qv = self.model.encode([f"query: {q}" for q in queries], normalize_embeddings=True)
        dense = (self.emb @ qv.T).max(axis=1)  # best match over the plain and follow-up queries

        if config.HYBRID_SEARCH:
            kw = self.bm25.scores(tokenize(question))
            if previous:
                kw = kw + 0.5 * self.bm25.scores(tokenize(previous))
            fused = rrf_ranks(dense) + rrf_ranks(kw, only_positive=True)
        else:
            fused = dense
        top = np.argsort(-fused)[:k]
        return [{**self.chunks[i], "score": float(dense[i])} for i in top]

    # ---------- answering ----------
    def prepare(self, question: str, k: int = config.TOP_K, history: list | None = None):
        """Search and build the LLM messages. Returns (messages, hits); messages is None when
        no slide is close enough, so the caller should refuse without calling the LLM.
        history = earlier chat as [{"role": "user"|"assistant", "content": ...}]."""
        history = [m for m in (history or []) if m.get("content")][-4:]  # last 2 exchanges
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        hits = self.retrieve(question, k, previous)
        if not hits or max(h["score"] for h in hits) < config.MIN_SCORE:
            return None, hits

        context = "\n\n".join(f"[{h['cite']}] ({h['title']})\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"][:800]} for m in history]
        messages.append({"role": "user",
                         "content": f"Slide excerpts:\n{context}\n\nQuestion: {question}"})
        return messages, hits

    def answer(self, question: str, k: int = config.TOP_K, history: list | None = None):
        """Full answer in one go (terminal + evaluation). Returns (text, hits)."""
        self.last_model = None
        messages, hits = self.prepare(question, k, history)
        if messages is None:
            return config.REFUSAL, hits
        reply = self.chat(messages)
        return (reply if reply is not None else config.BUSY_MESSAGE), hits

    # ---------- LLM calls with fallback ----------
    def _models(self) -> list:
        models = list(dict.fromkeys([config.LLM_MODEL] + config.FALLBACK_MODELS))
        ready = [m for m in models if _cooldown_until.get(m, 0) <= time.time()]
        return ready or models  # everything cooling down: try them all anyway

    def _failed(self, model: str, e: Exception):
        status = getattr(e, "status_code", "timeout/connection")
        print(f"[warn] {model} failed ({status}), trying next model...")
        if isinstance(e, APIStatusError) and e.status_code in (400, 401, 403):
            raise e  # bad request or bad key: other models won't help
        # A rate limit (429) clears within seconds; a timeout or 503 means the model is overloaded.
        pause = 15 if status == 429 else config.COOLDOWN_SECONDS
        _cooldown_until[model] = time.time() + pause

    def chat(self, messages: list, **kwargs):
        """Call the main model, falling back to the others. Returns None if all are busy.
        kwargs go to the API (e.g. temperature, response_format, timeout)."""
        kwargs.setdefault("temperature", 0)
        for model in self._models():
            try:
                resp = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
                self.last_model = model
                return (resp.choices[0].message.content or "").strip()
            except (APIStatusError, APIConnectionError) as e:
                self._failed(model, e)
        return None

    def stream(self, messages: list):
        """Like chat(), but yields the answer piece by piece as it is written.
        Falls back to the next model if one fails before writing anything."""
        self.last_model = None
        for model in self._models():
            try:
                chunks = iter(self.client.chat.completions.create(
                    model=model, messages=messages, temperature=0, stream=True))
                first = None
                for chunk in chunks:  # wait for the first text, so early failures fall back
                    if chunk.choices and chunk.choices[0].delta.content:
                        first = chunk.choices[0].delta.content
                        break
            except (APIStatusError, APIConnectionError) as e:
                self._failed(model, e)
                continue
            self.last_model = model
            if first:
                yield first
            try:
                for chunk in chunks:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except (APIStatusError, APIConnectionError):
                yield "\n\n*(The AI service stopped responding, so this answer is cut off. Please ask again.)*"
            return
        yield config.BUSY_MESSAGE

    # ---------- practice questions ----------
    def quiz_slides(self, lectures=None, topic: str = "", n: int = 12) -> list:
        """Pick the slides to write questions from: spread over the whole lecture, or the
        slides closest to `topic`. Random, so 'New questions' gives different ones."""
        pool = [c for c in self.chunks if (not lectures or c["lecture"] in lectures)
                and len(c["text"]) >= config.MIN_CHARS]
        if len(pool) <= n:
            return pool
        if topic.strip():
            order = {(c["pdf"], c["cite"]): i for i, c in enumerate(pool)}
            ranked = [h for h in self.retrieve(topic, k=len(self.chunks))
                      if (h["pdf"], h["cite"]) in order][:2 * n]
            picked = random.sample(ranked, min(n, len(ranked)))
            return sorted(picked, key=lambda c: order[(c["pdf"], c["cite"])])
        # Even coverage: split the lecture into n parts and take one random slide from each.
        parts = np.array_split(np.arange(len(pool)), n)
        return [pool[random.choice(list(p))] for p in parts if len(p)]

    def generate_quiz(self, n: int = 5, kind: str = "mixed", lectures=None, topic: str = ""):
        """Practice questions from the slides. Returns a list of question dicts, each with the
        slide it came from under "hit"; [] if the model gave nothing usable; None if busy."""
        slides = self.quiz_slides(lectures, topic, n=min(max(n + 3, 8), config.QUIZ_MAX_SLIDES))
        if not slides:
            return []
        context = "\n\n".join(f"[{c['cite']}] ({c['title']})\n{c['text']}" for c in slides)
        focus = f"\nFocus on this topic: {topic.strip()}" if topic.strip() else ""
        extra = max(3, n // 3)  # spare questions: a few usually fail the checks in clean_quiz
        messages = [
            {"role": "system", "content": QUIZ_PROMPT.format(n=n + extra, mix=QUIZ_MIX[kind])},
            {"role": "user", "content": f"Slide excerpts:\n{context}{focus}"},
        ]
        reply = self.chat(messages, temperature=0.8, timeout=config.QUIZ_TIMEOUT,
                          response_format={"type": "json_object"})
        if reply is None:
            return None
        return clean_quiz(reply, slides, kind)[:n]


def clean_quiz(reply: str, slides: list, kind: str = "mixed") -> list:
    """Parse the model's JSON and keep only well-formed questions that cite a real slide."""
    text = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return []
    items = data.get("questions", []) if isinstance(data, dict) else []

    by_cite = {c["cite"]: c for c in slides}
    out, seen = [], set()
    for q in items if isinstance(items, list) else []:
        if not isinstance(q, dict):
            continue
        qtype, question = q.get("type"), str(q.get("question", "")).strip()
        if qtype not in ("mcq", "short") or not question or question.lower() in seen:
            continue
        if kind != "mixed" and qtype != kind:
            continue
        hit = find_slide(str(q.get("cite", "")), by_cite, slides)
        if hit is None:
            continue  # cites a slide it wasn't given: likely made up
        item = {"type": qtype, "question": question, "hit": hit,
                "explanation": str(q.get("explanation", "")).strip()}
        if qtype == "mcq":
            opts = q.get("options")
            ans = q.get("answer")
            if (not isinstance(opts, list) or not 3 <= len(opts) <= 5
                    or not isinstance(ans, int) or not 0 <= ans < len(opts)):
                continue
            opts = [str(o).strip() for o in opts]
            if len(set(opts)) != len(opts) or not all(opts):
                continue
            # Models tend to put the right answer in the middle (B/C); shuffle so it can be anywhere.
            correct = opts[ans]
            random.shuffle(opts)
            item.update(options=opts, answer=opts.index(correct))
        else:
            ans = str(q.get("answer", "")).strip()
            if not ans:
                continue
            item["answer"] = ans
        seen.add(question.lower())
        out.append(item)
    return out


def find_slide(cite: str, by_cite: dict, slides: list):
    """Match a cite tag to one of the given slides; tolerates 'p.14' inside a 'p.13-14' chunk."""
    cite = cite.strip().strip("[]")
    if cite in by_cite:
        return by_cite[cite]
    m = re.match(r"(\S+) p\.(\d+)", cite)
    if m:
        for c in slides:
            if c["lecture"] == m.group(1) and int(m.group(2)) in c["pages"]:
                return c
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
