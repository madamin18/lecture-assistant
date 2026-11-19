# Lecture Notes Assistant (RAG over course slides)

Live site: https://lecture-assistant-madamin.streamlit.app

Preloaded with all 9 lectures of *Design of Generative AI Application Systems* (491 slides):

| | Lecture | | Lecture |
|---|---|---|---|
| L01 | Generative AI Systems: Uses, Prompts, and Limits | L06 | RAG Evaluation: Test Sets and Failure Analysis |
| L02 | LLM Essentials: Tokens, Context, Temperature, Embeddings | L07 | Generative AI in Industry (special lecture) |
| L03 | RAG I: Ingestion, Chunking, and Vector Indexing | L08 | Agentic Workflows: LangGraph and CrewAI |
| L04 | RAG II: Retrieval and Grounded Generation | L09 | Self-improving Agents and Multi-agent RAG |
| L05 | Improving RAG: Rewriting, Filtering, and Relevance | | |

Only the slide PDFs are published; the lab handouts, readings and quiz keys from the course
packages are not.

Three tabs:
- **Ask** (English or 한국어): chat with the lecture slides. Citations appear as badges in the
  answer. Each answer shows how it was found (translation, hybrid search, which model answered,
  timings), the cited slides as images, a relevance table of the other slides searched, thumbs
  up/down, and "Ask next" follow-ups named after the titles of the other slides found. The
  sidebar can limit the search to chosen lectures, and "Save chat" downloads the chat as
  Markdown study notes with the slides each answer cites.
- **Practice exam**: exam-style questions (multiple choice and/or short answer) from the slides,
  in English, Korean or the slides' language. Picking an answer shows right away whether it is
  correct, with an explanation and the source slide; short answers reveal a model answer and
  the student grades themselves. A missed question has an "Explain this to me" button that asks
  the assistant about it (searching only that lecture). A live score and progress bar, balloons
  at 80%+, "Retry the ones I missed", "Try again", and a printable study sheet (questions first,
  answer key at the end).
- **Browse lectures**: a card per lecture with its cover slide, a slide viewer whose slider shows
  each slide's title, and shortcuts to ask about a slide, search one lecture, or get a
  5-question quiz on it.

Styling lives in `.streamlit/config.toml` (light and dark theme, fonts, radii); no custom CSS.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # then put your API key in .env
```
Put the lecture PDFs in `data/slides/`, named like `L03_RAG_I_Ingestion_and_Indexing.pdf`.
The `L03` part becomes the start of the citation tag (`[L03 slide 14]`) and the rest becomes the
lecture title shown in the app.

## Run
```bash
python ingest.py                     # build the index (re-run whenever slides change)
python rag.py "What is RAG?"         # quick test in the terminal
python -m streamlit run app.py       # website at http://localhost:8501
python warm_cache.py                 # prepare instant answers for the demo questions
```
`warm_cache.py` stores answers for the starter questions and the demo path in
`index/answer_cache.json`; the site shows them instantly (and they still work if the AI service
is busy). They are ignored automatically once the index is rebuilt; re-run the script then. The
site also remembers every chat's first question while it runs, so repeated questions are
instant too. Follow-up questions are always answered fresh.

## How it works
1. **Ingest** (`ingest.py`): one chunk per slide; slides with under 80 characters (titles,
   diagrams) are merged into the next slide. Chunks are embedded with
   `intfloat/multilingual-e5-small` (free, runs locally, English + Korean). Each slide's title is
   read from the PDF as the largest text near the top (found for all 491 slides); the website
   uses it for follow-up suggestions, the slide viewer and the saved notes.
2. **Slide numbers**: citations use the number printed in each slide's footer, because exported
   decks often skip hidden slides (in L01, PDF page 56 is printed as slide 59). The right-most
   number in the bottom 15% of the page is used, and only if it is found on at least 70% of the
   pages and always increases; otherwise citations fall back to PDF pages (`[L01 p.56]`).
   A missing number is filled in only when its neighbours leave one possibility (an unnumbered
   title page before slide 2 is slide 1). A number right after "Lesson"/"Week" inside the footer
   text is skipped, so "Lesson 7 / Testing" on an unnumbered slide isn't read as slide 7, while
   L07's footer "Lesson 07 | Industry special lecture 12" still gives slide 12. Slide images are captioned with both numbers, e.g.
   "Generative AI Systems · slide 59 (PDF page 56)".
3. **Search** (`rag.py`): hybrid search. Meaning-based search (embeddings) and keyword search
   (BM25) are combined with reciprocal rank fusion. Keyword search finds slides that share a rare
   exact word with the question, e.g. "submit" -> the *Submit evidence* slide, which
   meaning-based search ranked 10th. A question that names a lecture ("the Lesson 1 lab",
   "lecture 3", "L05", "3강", "3차시") searches only that lecture; several lectures have a lab,
   so "What must students submit for the lab?" is otherwise ambiguous.
4. **Korean questions about English slides** (or the reverse): the question is first translated
   into the slides' language with one short LLM call, and both versions are searched.
   Without this, half of the Korean test questions found none of the right slides and short ones
   such as "환각이란 무엇인가요?" scored below the refusal threshold. Answers come back in the
   question's language with key terms from the slides in parentheses, e.g. 환각(hallucination).
5. **Refusal**: if no slide is similar enough (`MIN_SCORE`), the app replies "This isn't covered
   in the lecture slides I have." (or 강의 슬라이드에서 다루지 않는 내용입니다.) without calling
   the LLM. Otherwise the prompt also tells the model to refuse when the excerpts don't contain
   the answer.
6. **Answer**: the top 8 slides (`TOP_K`) go to the LLM with rules to cite every sentence. The answer is
   streamed (text appears as it's written). The last two exchanges are included, so follow-ups
   such as "can you give an example of that?" work.
7. **Practice questions**: slides are sampled evenly across the lecture (or the slides closest to
   the chosen topic), the LLM writes questions as JSON, and every question is checked: it must
   cite a slide it was given, multiple choice needs 4 distinct options with one correct answer,
   and options are shuffled so the right answer isn't always B or C.

## Models and reliability
- Default: `gemini-3.5-flash-lite` (answers in 1-3 s); fallbacks `gemini-3-flash-preview`,
  `gemini-3.1-flash-lite`, `gemini-3.5-flash`. In testing (Nov 2026) `gemini-3.5-flash` was often
  overloaded (timeouts, 503) and `gemini-flash-latest` / `3.7-flash` / `3.8-flash` timed out.
- **Free-tier quotas are per model and per day** (500 requests for the flash-lite models, 20 for
  `gemini-3.5-flash`; they reset at midnight Pacific time, 17:00 KST). A full graded evaluation
  run uses about 140 requests, so three runs in one day used up `gemini-3.1-flash-lite`. Listing
  several models multiplies the daily budget, and a model whose daily quota is used up is skipped
  for an hour (the app reads the `retryDelay` in the error) instead of being retried.
- A model that times out (`LLM_TIMEOUT` = 20 s) or is overloaded (503) is skipped for
  `COOLDOWN_SECONDS` = 120 s; a per-minute rate limit (429) for 15 s. If every model fails
  quickly, the app waits 2 s and tries once more before saying the service is busy.
- A model that returns an empty reply (e.g. blocked by a safety filter) is treated like a failure,
  so the next model answers instead of the student getting a blank answer.
- The app reloads `rag.py`, `config.py`, `ingest.py`, the demo index and the saved answers when
  they change, so a deploy that only touches those files takes effect without rebooting the
  server.
- All visitors share one search object, so it keeps no per-question state: each search fills in
  its own `info` dict (translation, lectures searched, model), and two people asking at once
  can't see each other's "How I found this" steps.
- `python check_models.py` shows which models currently answer with your key.

## Tests
```bash
pip install pytest
python -m pytest -q
```
47 tests in `tests/` cover keyword search, lecture mentions, language detection, slide titles and
footer numbers (on PDFs generated in the test), merging, the lecture filter, translation of Korean
questions, refusals, model fallback and cooldowns, streaming, practice-question checks and the
saved-answer cache. They use a fake embedder and a fake AI client, so they run in under a second
with no API key, no internet and no PyTorch. GitHub Actions runs them on every push
(`.github/workflows/tests.yml`).

## How the website handles uploads
- Visitors upload their own lecture PDFs in the sidebar. Files are indexed in memory for that
  browser tab only: no one else can search them and nothing is saved on the server.
- With no uploads, the site runs in **demo mode** on the slides in `data/slides/` (index built by
  `ingest.py`): the 9 course lectures, with one-click example questions (including a Korean
  one).
- Limits: 50 MB per file (`.streamlit/config.toml`), 300 pages per visitor
  (`MAX_UPLOAD_PAGES` in `config.py`). Scanned PDFs without a text layer can't be searched.
- Citation tags start with the file name: `L03_RAG_basics.pdf` -> `[L03 slide 14]`; any other
  name is used as is, e.g. `Week 3.pdf` -> `[Week_3 slide 14]`.

## Evaluate
`eval/test_set.jsonl` has 47 questions covering all 9 lectures: 40 answerable (direct,
paraphrased, multi-slide; 29 in English, 11 in Korean) and 7 out of scope (4 English, 3 Korean).
Each answerable question lists every PDF page that answers it ("relevance judgments", as in
Lesson 6). When the other lectures were added, three judgments were widened after reading the
slides that outranked the expected ones (e.g. L04 slide 13: "A similarity score helps rank
candidates. Source text establishes claim support."), q18 (attention) moved from out of scope to
answerable because L02 slide 26 covers it, and the lab-submission questions (q13, q24) now name
Lesson 1 because four lectures have labs.
```bash
python evaluate.py --retrieval-only               # search only (API used only to translate)
python evaluate.py --retrieval-only --no-translate --no-hybrid   # older search, no API calls
python evaluate.py --name full_course --judge --delay 4   # full run; the LLM grades answers
```
Options: `--delay N` waits between questions (avoids free-tier 429 errors),
`--min-score 0.8` tries a different refusal threshold, `--no-hybrid` turns keyword search off,
`--no-translate` turns question translation off. Questions where every model was busy are
reported as `n_busy_skipped` and left out of the percentages, so re-run if that number is high.
Results, with an English/Korean breakdown: `eval/summary_*.json`.

**Whole course** (47 questions, `gemini-3.1-flash-lite`):

| | 5 slides per answer | 8 slides per answer (now the default) |
|---|---|---|
| Questions scored | 47 | 38 (the last 9 hit the daily API quota) |
| Retrieval hit@k | 97.5% | 100% |
| Citation accuracy | 95.0% | 96.9% |
| False refusals (answerable but refused) | 5.0% | 0% |
| Correct refusals (out of scope) | 85.7% (6/7) | 83.3% (5/6) |
| Answers graded correct | 90.0% (+5.0% partial) | 100% |

With 8x more slides, more slides from other lectures compete for the top 5; with 8 slides the
model had enough context for every question it was asked.

**Lecture 1 only** (earlier rounds, before the other lectures were added):

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

The missed refusal (q19, Mobility Office weekend hours) is a near miss in every run: the slides
use the Mobility Office as an example, and the model replies that the slides don't support any
weekend hours, just not with the exact refusal sentence the evaluation checks for. q13 ("What
must students submit for the lab?") went from a retrieval miss to a correct answer thanks to
keyword search.

### Practice question quality
`python eval_quiz.py` generates a 5-question quiz for every lecture plus one in Korean (50
questions) and checks each against the slide it cites (results: `eval/quiz_quality.json`,
every question: `eval/quiz_results.jsonl`).

| Check (50 questions, Nov 18) | Result |
|---|---|
| Answerable from the cited slide | 100% |
| Marked answer correct | 100% |
| Multiple choice with exactly one correct option (41 questions) | 100% |
| Clear and well formed | 100% |
| Correct option position after shuffling (A/B/C/D) | 8 / 12 / 10 / 11 |
| Average time per 5-question quiz | 7.8 s |

The checker is the same model family as the generator, so these are estimates; a manual check
of 6 random questions agreed. Most questions test recall of a slide, and some wrong options are
too easy, which the checks above don't measure.

## Deploy
1. Push this folder to GitHub, **including `index/` and `data/slides/`**.
2. On share.streamlit.io: Create app -> pick the repo -> main file `app.py`.
3. In the app's Settings -> Secrets:
   ```toml
   GEMINI_API_KEY = "your-key"
   LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
   LLM_MODEL = "gemini-3.5-flash-lite"
   ```
   Every push to `main` redeploys the site.
