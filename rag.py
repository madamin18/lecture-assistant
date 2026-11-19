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

import config
from ingest import open_pdf

SYSTEM_PROMPT = """You are a study assistant for a university course.
Answer the student's question using ONLY the lecture slide excerpts provided.

Rules:
- After every sentence that uses a slide, cite it with its tag in square brackets,
  copied exactly from the excerpt headers, e.g. [L03 slide 14]. One tag per bracket.
- If the excerpts do not contain the answer, reply with exactly:
  "{refusal}"
- Never use outside knowledge, and never invent slide numbers.
- The excerpts can come from several lectures (the tag starts with the lecture, e.g. L03).
  If the question could mean more than one of them (e.g. "the lab" when several lectures have
  a lab), answer briefly for each lecture the excerpts cover and name it, or ask which lecture
  the student means. Don't refuse only because the question is ambiguous.
- Earlier messages in the chat are only there so you know what a follow-up question
  refers to ("it", "that", "explain more"). The facts must still come from the excerpts.
- Be concise. Answer in the same language as the question. If the slides are in another
  language, keep key technical terms from the slides in parentheses, e.g. 환각(hallucination)."""

TRANSLATE_PROMPT = """Translate the student's question into {language} so it can be used to search
lecture slides written in {language}. Keep technical terms. If it is a follow-up that refers to
the previous question ("that", "it", "more"), write it as a standalone question.
Reply with the translated question only."""

LANGUAGE_NAMES = {"en": "English", "ko": "Korean"}

QUIZ_PROMPT = """You write exam practice questions for university students, using ONLY the
lecture slide excerpts provided. Every question must be answerable from the slide it cites.

Write {n} questions. {mix}
Return JSON only, in exactly this shape:
{{"questions": [
  {{"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."], "answer": 0,
    "explanation": "why the answer is right, in 1-2 sentences", "cite": "L01 slide 14"}},
  {{"type": "short", "question": "...", "answer": "a model answer in 1-3 sentences",
    "explanation": "what a full-mark answer must mention", "cite": "L01 slide 14"}}
]}}

Rules:
- "mcq" has exactly 4 options and exactly one correct option; "answer" is its index (0-3).
  Wrong options must be plausible to a student who skimmed the slides. Avoid "all of the
  above" / "none of the above" (the options get shuffled).
- "cite" is the tag of the one slide the answer comes from, copied exactly from the excerpts.
- Test understanding: definitions, but also why, compare, and apply-to-an-example questions.
- Cover different slides; never ask two questions about the same fact.
- Skip slide footers, page numbers, layout, and title or agenda slides.
- {language}"""

QUIZ_LANGUAGE = {
    "same": "Write everything in the same language as the slides.",
    "en": "Write everything (questions, options, answers, explanations) in English.",
    "ko": "Write everything (questions, options, answers, explanations) in Korean. Keep key "
          "technical terms from the slides in parentheses, e.g. 환각(hallucination).",
}

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


def language(text: str) -> str:
    """'ko' if most letters are Hangul, else 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    hangul = sum("\uac00" <= ch <= "\ud7a3" for ch in letters)
    return "ko" if letters and hangul / len(letters) > 0.3 else "en"


def refusal_for(question: str) -> str:
    return config.REFUSAL_KO if language(question) == "ko" else config.REFUSAL


# "Lesson 1", "lecture 03", "L05", "1강", "3차시". A bare "L" needs two digits, so terms like
# "L2 norm" aren't read as lecture 2.
LECTURE_MENTION = re.compile(
    r"\b(?:lesson|lecture|lec)\s*0?(\d{1,2})\b|\bL(\d{2})\b|(\d{1,2})\s*(?:강|차시)", re.IGNORECASE)


def mentioned_lectures(text: str, available) -> set:
    """Lecture tags the question names (e.g. 'the Lesson 1 lab' -> {'L01'}), if they exist."""
    found = set()
    for m in LECTURE_MENTION.finditer(text):
        tag = f"L{int(next(g for g in m.groups() if g)):02d}"
        if tag in available:
            found.add(tag)
    return found


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


def retry_delay(e: Exception) -> float | None:
    """Seconds the API asks us to wait (Gemini puts "retryDelay": "76943s" in its 429 body)."""
    response = getattr(e, "response", None)
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', getattr(response, "text", "") or "")
    return float(m.group(1)) if m else None


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

    The web app shares one SlideRAG between all visitors, so it keeps no per-question state.
    Methods that the UI explains ("How I found this") take an optional `info` dict and fill it
    in: translation, lectures searched, and the model that answered.
    """

    def __init__(self, index_dir: str = config.INDEX_DIR, chunks=None, emb=None,
                 pdf_data=None, embedder=None, client=None):
        if chunks is None:
            d = Path(index_dir)
            chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
            emb = np.load(d / "embeddings.npy")
        self.chunks, self.emb = chunks, emb
        self.bm25 = BM25([tokenize(f"{c['title']} {c['text']}") for c in chunks])
        self.slide_lang = language(" ".join(c["text"] for c in chunks[:50]))
        self.chunk_lectures = np.array([c["lecture"] for c in chunks])
        self.lecture_tags = set(self.chunk_lectures.tolist())
        self.pdf_data = pdf_data or {}  # uploaded file name -> PDF bytes
        if embedder is None:
            from sentence_transformers import SentenceTransformer  # slow import: only if needed
            embedder = SentenceTransformer(config.EMBED_MODEL)
        self.model = embedder
        self.client = client or make_client()

    @property
    def lectures(self) -> list:
        """[(lecture tag, title)] in index order, e.g. [('L01', 'Intro')]."""
        seen = {}
        for c in self.chunks:
            seen.setdefault(c["lecture"], c["title"])
        return list(seen.items())

    # ---------- search ----------
    def retrieve(self, question: str, k: int = config.TOP_K, previous: str | None = None,
                 translation: str | None = None, only=None, info: dict | None = None) -> list:
        """Top-k slides. `previous` = the student's previous question, used so follow-ups like
        "explain that more simply" still find the right slides. `translation` = the question in
        the slides' language (see translate()). `only` = lecture tags to search (e.g. a filter
        chosen in the UI); a lecture named in the question takes precedence.
        Each hit's "score" is the meaning-based (cosine) similarity, used for the refusal check."""
        info = {} if info is None else info
        if not self.chunks:
            return []
        queries = [question] + ([translation] if translation else []) \
            + ([f"{previous} {question}"] if previous else [])
        qv = self.model.encode([f"query: {q}" for q in queries], normalize_embeddings=True)
        dense = (self.emb @ qv.T).max(axis=1)  # best match over all phrasings

        if config.HYBRID_SEARCH:
            kw = self.bm25.scores(tokenize(question))
            if translation:
                kw = kw + self.bm25.scores(tokenize(translation))
            if previous:
                kw = kw + 0.5 * self.bm25.scores(tokenize(previous))
            fused = rrf_ranks(dense) + rrf_ranks(kw, only_positive=True)
        else:
            fused = dense
        # A question that names a lecture ("the Lesson 1 lab", "3강") searches only that lecture.
        named = mentioned_lectures(f"{question} {translation or ''}", self.lecture_tags)
        allowed = named or ({t for t in only if t in self.lecture_tags} if only else set())
        if allowed:
            fused = np.where(np.isin(self.chunk_lectures, list(allowed)), fused, -np.inf)
        info.update(lectures=sorted(allowed), named=bool(named))
        # isfinite: when the allowed lectures have fewer than k slides, don't pad with others
        top = [i for i in np.argsort(-fused)[:k] if np.isfinite(fused[i])]
        return [{**self.chunks[i], "score": float(dense[i])} for i in top]

    def search(self, question: str, k: int = config.TOP_K, previous: str | None = None,
               only=None, info: dict | None = None):
        """retrieve(), first translating the question if it isn't in the slides' language.
        Returns (hits, min_score): the similarity needed to answer instead of refusing."""
        info = {} if info is None else info
        translation = None
        cross_language = language(question) != self.slide_lang
        if cross_language and config.TRANSLATE_QUERIES:
            translation = self.translate(question, previous)
        info["translation"] = translation
        hits = self.retrieve(question, k, previous, translation, only, info)
        # Untranslated cross-language questions score lower, so don't refuse them on score alone;
        # the prompt's refusal rule still applies.
        return hits, (0 if cross_language and not translation else config.MIN_SCORE)

    # ---------- answering ----------
    def prepare(self, question: str, k: int = config.TOP_K, history: list | None = None,
                only=None, info: dict | None = None):
        """Search and build the LLM messages. Returns (messages, hits); messages is None when
        no slide is close enough, so the caller should refuse without calling the LLM.
        history = earlier chat as [{"role": "user"|"assistant", "content": ...}]."""
        history = [m for m in (history or []) if m.get("content")][-4:]  # last 2 exchanges
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        hits, min_score = self.search(question, k, previous, only, info)
        if not hits or max(h["score"] for h in hits) < min_score:
            return None, hits

        context = "\n\n".join(f"[{h['cite']}] ({h['title']})\n{h['text']}" for h in hits)
        messages = [{"role": "system",
                     "content": SYSTEM_PROMPT.format(refusal=refusal_for(question))}]
        messages += [{"role": m["role"], "content": m["content"][:800]} for m in history]
        # Name the answer language: after a Korean chat, an English question was answered in Korean.
        messages.append({"role": "user",
                         "content": f"Slide excerpts:\n{context}\n\nQuestion: {question}\n\n"
                                    f"(Answer in {LANGUAGE_NAMES[language(question)]}.)"})
        return messages, hits

    def answer(self, question: str, k: int = config.TOP_K, history: list | None = None,
               info: dict | None = None):
        """Full answer in one go (terminal + evaluation). Returns (text, hits).
        info["model"] is the model that answered (None: refused without asking the LLM)."""
        info = {} if info is None else info
        info["model"] = None
        messages, hits = self.prepare(question, k, history, info=info)
        if messages is None:
            return refusal_for(question), hits
        reply = self.chat(messages, info=info)
        return (reply if reply is not None else config.BUSY_MESSAGE), hits

    def translate(self, question: str, previous: str | None = None) -> str | None:
        """The question in the slides' language, for searching. None if the service is busy."""
        prompt = TRANSLATE_PROMPT.format(language=LANGUAGE_NAMES[self.slide_lang])
        text = f"Previous question: {previous}\nQuestion: {question}" if previous else question
        reply = self.chat([{"role": "system", "content": prompt},
                           {"role": "user", "content": text}], timeout=10)
        return reply or None

    # ---------- LLM calls with fallback ----------
    def _attempts(self):
        """Yields the models to try, in order. If every model failed quickly (rate limit or
        'overloaded' replies, not slow timeouts), waits 2 s and tries them all once more, since
        those errors often clear within seconds."""
        models = list(dict.fromkeys([config.LLM_MODEL] + config.FALLBACK_MODELS))
        ready = [m for m in models if _cooldown_until.get(m, 0) <= time.time()]
        start = time.time()
        yield from ready or models  # everything cooling down: try them all anyway
        if time.time() - start < 10:
            time.sleep(2)
            # soonest-available first: a rate-limited model before an overloaded one
            yield from sorted(models, key=lambda m: _cooldown_until.get(m, 0))

    def _failed(self, model: str, e: Exception):
        status = getattr(e, "status_code", "timeout/connection")
        print(f"[warn] {model} failed ({status}), trying next model...")
        if isinstance(e, APIStatusError) and e.status_code in (400, 401, 403):
            raise e  # bad request or bad key: other models won't help
        if status == 429:
            # A per-minute rate limit clears within seconds. A used-up DAILY quota comes with a
            # long retryDelay: skip the model for up to an hour instead of retrying it all day.
            wait = retry_delay(e)
            pause = min(wait, 3600) if wait and wait > 60 else 15
        else:  # a timeout or 503 means the model is overloaded
            pause = config.COOLDOWN_SECONDS
        _cooldown_until[model] = time.time() + pause

    def chat(self, messages: list, info: dict | None = None, **kwargs):
        """Call the main model, falling back to the others. Returns None if all are busy.
        Sets info["model"] to the model that answered. kwargs go to the API (e.g. temperature,
        response_format, timeout)."""
        kwargs.setdefault("temperature", 0)
        for model in self._attempts():
            try:
                resp = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
            except (APIStatusError, APIConnectionError) as e:
                self._failed(model, e)
                continue
            text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            if not text:  # empty reply (e.g. blocked by a safety filter): try the next model
                print(f"[warn] {model} returned an empty reply, trying next model...")
                continue
            if info is not None:
                info["model"] = model
            return text
        return None

    def stream(self, messages: list, info: dict | None = None):
        """Like chat(), but yields the answer piece by piece as it is written.
        Falls back to the next model if one fails (or stays silent) before writing anything.
        Sets info["model"] once a model starts answering."""
        for model in self._attempts():
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
            if first is None:
                print(f"[warn] {model} returned an empty reply, trying next model...")
                continue
            if info is not None:
                info["model"] = model
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
            ranked = [h for h in self.search(topic, k=len(self.chunks))[0]  # translates if needed
                      if (h["pdf"], h["cite"]) in order][:2 * n]
            picked = random.sample(ranked, min(n, len(ranked)))
            return sorted(picked, key=lambda c: order[(c["pdf"], c["cite"])])
        # Even coverage: split the lecture into n parts and take one random slide from each.
        parts = np.array_split(np.arange(len(pool)), n)
        return [pool[random.choice(list(p))] for p in parts if len(p)]

    def generate_quiz(self, n: int = 5, kind: str = "mixed", lectures=None, topic: str = "",
                      lang: str = "same", info: dict | None = None):
        """Practice questions from the slides. Returns a list of question dicts, each with the
        slide it came from under "hit"; [] if the model gave nothing usable; None if busy."""
        slides = self.quiz_slides(lectures, topic, n=min(max(n + 3, 8), config.QUIZ_MAX_SLIDES))
        if not slides:
            return []
        context = "\n\n".join(f"[{c['cite']}] ({c['title']})\n{c['text']}" for c in slides)
        focus = f"\nFocus on this topic: {topic.strip()}" if topic.strip() else ""
        extra = max(3, n // 3)  # spare questions: a few usually fail the checks in clean_quiz
        messages = [
            {"role": "system", "content": QUIZ_PROMPT.format(
                n=n + extra, mix=QUIZ_MIX[kind], language=QUIZ_LANGUAGE[lang])},
            {"role": "user", "content": f"Slide excerpts:\n{context}{focus}"},
        ]
        reply = self.chat(messages, info, temperature=0.8, timeout=config.QUIZ_TIMEOUT,
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
    """Match a cite tag to one of the given slides; tolerates 'slide 14' inside a
    'slide 13-14' chunk."""
    cite = cite.strip().strip("[]")
    if cite in by_cite:
        return by_cite[cite]
    m = re.match(r"(\S+) (slide |p\.)(\d+)", cite)
    if m:
        lec, kind, num = m.group(1), m.group(2), int(m.group(3))
        for c in slides:
            nums = (c.get("labels") or []) if kind == "slide " else c["pages"]
            if c["lecture"] == lec and num in nums:
                return c
    return None


def cited_hits(answer: str, hits: list) -> list:
    """The retrieved slides the answer actually cites, in the order they were retrieved."""
    # (?![\d-]) stops "L01 p.1" from matching inside "L01 p.13" or "L01 p.1-2"
    return [h for h in hits if re.search(re.escape(h["cite"]) + r"(?![\d-])", answer)]


# "[L03 slide 14]", "[L03 p.5-6]", and a few tags in one bracket: "[L01 slide 3; L01 slide 9]"
_TAG = r"[^\[\]\n;,]+? (?:slide |p\.)\d+(?:-\d+)?"
CITATION = re.compile(rf"\[({_TAG}(?:\s*[;,]\s*{_TAG})*)\]")


def badge_citations(answer: str) -> str:
    """Markdown with each citation drawn as a small badge (Streamlit's :color-badge[...])."""
    return CITATION.sub(lambda m: " ".join(
        f":violet-badge[{t.strip()}]" for t in re.split(r"[;,]", m.group(1))), answer)


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
    info = {}
    ans, hits = rag.answer(" ".join(sys.argv[1:]) or "What is retrieval-augmented generation?",
                           info=info)
    print(f"(model: {info['model']})\n")
    print(ans, "\n\nRetrieved:")
    for h in hits:
        print(f"  [{h['cite']}] score={h['score']:.3f}")
