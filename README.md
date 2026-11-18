# Lecture Notes Assistant (RAG over course slides)

Live site: https://lecture-assistant-madamin.streamlit.app

Two modes:
- **Ask a question** (English or 한국어): chat with the lecture slides. Every answer cites slides
  like `[L01 slide 14]` and shows the slide images, so students can check the source.
- **Practice exam**: generates exam-style questions (multiple choice and/or short answer) from
  the slides, in English, Korean or the slides' language. Students answer, click
  **Check answers**, and see the correct answer, an explanation and the slide it came from.
  Questions can focus on a topic and be downloaded as a printable study sheet (questions first,
  answer key at the end).

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # then put your API key in .env
```
Put the lecture PDFs in `data/slides/`, named like `L01_Intro.pdf`, `L03_RAG_basics.pdf`.
The `L01` part becomes the start of the citation tag, e.g. `[L03 slide 14]`.

## Run
```bash
python ingest.py                     # build the index (re-run whenever slides change)
python rag.py "What is RAG?"         # quick test in the terminal
python -m streamlit run app.py       # website at http://localhost:8501
```

## How it works
1. **Ingest** (`ingest.py`): one chunk per slide; slides with under 80 characters (titles,
   diagrams) are merged into the next slide. Chunks are embedded with
   `intfloat/multilingual-e5-small` (free, runs locally, English + Korean).
2. **Slide numbers**: citations use the number printed in each slide's footer, because exported
   decks often skip hidden slides (in L01, PDF page 56 is printed as slide 59). The right-most
   number in the bottom 15% of the page is used, and only if it is found on at least 70% of the
   pages and always increases; otherwise citations fall back to PDF pages (`[L01 p.56]`).
   Slide images are captioned with both, e.g. "Intro · slide 59 (PDF page 56)".
3. **Search** (`rag.py`): hybrid search. Meaning-based search (embeddings) and keyword search
   (BM25) are combined with reciprocal rank fusion. Keyword search finds slides that share a rare
   exact word with the question, e.g. "submit" -> the *Submit evidence* slide, which
   meaning-based search ranked 10th.
4. **Korean questions about English slides** (or the reverse): the question is first translated
   into the slides' language with one short LLM call, and both versions are searched.
   Without this, half of the Korean test questions found none of the right slides and short ones
   such as "환각이란 무엇인가요?" scored below the refusal threshold. Answers come back in the
   question's language with key terms from the slides in parentheses, e.g. 환각(hallucination).
5. **Refusal**: if no slide is similar enough (`MIN_SCORE`), the app replies "This isn't covered
   in the lecture slides I have." (or 강의 슬라이드에서 다루지 않는 내용입니다.) without calling
   the LLM. Otherwise the prompt also tells the model to refuse when the excerpts don't contain
   the answer.
6. **Answer**: the top 5 slides go to the LLM with rules to cite every sentence. The answer is
   streamed (text appears as it's written). The last two exchanges are included, so follow-ups
   such as "can you give an example of that?" work.
7. **Practice questions**: slides are sampled evenly across the lecture (or the slides closest to
   the chosen topic), the LLM writes questions as JSON, and every question is checked: it must
   cite a slide it was given, multiple choice needs 4 distinct options with one correct answer,
   and options are shuffled so the right answer isn't always B or C.

## Models and reliability
- Default: `gemini-3.1-flash-lite` (answers in 1-3 s), fallback `gemini-3.5-flash`.
  In testing (Nov 2026) `gemini-3.5-flash` was often overloaded (timeouts, 503) and
  `gemini-flash-latest` never answered.
- A model that times out (`LLM_TIMEOUT` = 20 s) or is overloaded (503) is skipped for
  `COOLDOWN_SECONDS` = 120 s; a rate-limited one (429) for 15 s. If every model fails quickly,
  the app waits 2 s and tries once more before saying the service is busy.
- The app reloads `rag.py`, `config.py` and `ingest.py` when they change, so a deploy that only
  touches those files takes effect without rebooting the server.
- `python check_models.py` shows which models currently answer with your key.

## How the website handles uploads
- Visitors upload their own lecture PDFs in the sidebar. Files are indexed in memory for that
  browser tab only: no one else can search them and nothing is saved on the server.
- With no uploads, the site runs in **demo mode** on the slides in `data/slides/` (index built by
  `ingest.py`), with one-click example questions (including a Korean one).
- Limits: 50 MB per file (`.streamlit/config.toml`), 300 pages per visitor
  (`MAX_UPLOAD_PAGES` in `config.py`). Scanned PDFs without a text layer can't be searched.
- Citation tags start with the file name: `L03_RAG_basics.pdf` -> `[L03 slide 14]`; any other
  name is used as is, e.g. `Week 3.pdf` -> `[Week_3 slide 14]`.

## Evaluate
`eval/test_set.jsonl` has 28 questions: 21 answerable (direct, paraphrased, multi-slide; 13 in
English, 8 in Korean) and 7 out of scope (4 English, 3 Korean). Each answerable question lists
the PDF pages that contain the answer.
```bash
python evaluate.py --retrieval-only               # search only (API used only to translate)
python evaluate.py --retrieval-only --no-translate --no-hybrid   # older search, no API calls
python evaluate.py --name improved_v2 --judge --delay 4   # full run; the LLM grades answers
```
Options: `--delay N` waits between questions (avoids free-tier 429 errors),
`--min-score 0.8` tries a different refusal threshold, `--no-hybrid` turns keyword search off,
`--no-translate` turns question translation off. Questions where every model was busy are
reported as `n_busy_skipped` and left out of the percentages, so re-run if that number is high.
Results, with an English/Korean breakdown: `eval/summary_*.json`.

| | baseline | improved_v1 | improved_v2 |
|---|---|---|---|
| Test questions | 20 | 20 | 28 (8 more in Korean) |
| Search | meaning only | + keyword (hybrid) | + translation of Korean questions |
| Model | gemini-flash-latest -> 3.5-flash | 3.1-flash-lite | 3.1-flash-lite |
| Retrieval hit@5 | 93.3% | 100% | 100% |
| Citation accuracy | 93.3% | 100% | 100% |
| Correct refusals (out of scope) | 80.0% | 80.0% | 85.7% (6/7) |
| Answers graded correct | 80.0% | 93.3% | 100% |
| Time per answer | ~4 min (main model hung) | ~1-2 s | ~1-3 s |

Korean questions only (8 answerable, `--retrieval-only`): the right slide was in the top 5 for
50% of them without translation and 100% with it.

The one missed refusal (q19, Mobility Office weekend hours) is a near miss: the slides use the
Mobility Office as an example, and the model replied that the slides don't support any weekend
hours, just not with the exact refusal sentence the evaluation checks for. q13 ("What must
students submit for the lab?") went from a retrieval miss to a correct answer thanks to keyword
search.

## Deploy
1. Push this folder to GitHub, **including `index/` and `data/slides/`**.
2. On share.streamlit.io: Create app -> pick the repo -> main file `app.py`.
3. In the app's Settings -> Secrets:
   ```toml
   GEMINI_API_KEY = "your-key"
   LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
   LLM_MODEL = "gemini-3.1-flash-lite"
   ```
   Every push to `main` redeploys the site.
