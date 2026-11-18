# Lecture Notes Assistant (RAG over course slides)

Live site: https://lecture-assistant-madamin.streamlit.app

Two modes:
- **Ask a question**: chat with the lecture slides. Every answer cites slides like `[L01 p.14]`
  and shows the slide images, so students can check the source.
- **Practice exam**: generates exam-style questions (multiple choice and/or short answer) from
  the slides. Students answer, click **Check answers**, and see the correct answer, an
  explanation and the slide it came from. Questions can focus on a topic and be downloaded as a
  printable study sheet (questions first, answer key at the end).

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # then put your API key in .env
```
Put the lecture PDFs in `data/slides/`, named like `L01_Intro.pdf`, `L03_RAG_basics.pdf`.
The `L01` part becomes the citation tag, e.g. `[L03 p.14]`.

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
2. **Search** (`rag.py`): hybrid search. Meaning-based search (embeddings) and keyword search
   (BM25) are combined with reciprocal rank fusion. Keyword search finds slides that share a rare
   exact word with the question, e.g. "submit" -> the *Submit evidence* slide, which
   meaning-based search ranked 10th.
3. **Refusal**: if no slide is similar enough (`MIN_SCORE`), the app answers "This isn't covered
   in the lecture slides I have." without calling the LLM. Otherwise the prompt also tells the
   model to refuse when the excerpts don't contain the answer.
4. **Answer**: the top 5 slides go to the LLM with rules to cite every sentence. The answer is
   streamed (text appears as it's written). The last two exchanges are included, so follow-ups
   such as "can you give an example of that?" work.
5. **Practice questions**: slides are sampled evenly across the lecture (or the slides closest to
   the chosen topic), the LLM writes questions as JSON, and every question is checked: it must
   cite a slide it was given, multiple choice needs 4 distinct options with one correct answer,
   and options are shuffled so the right answer isn't always B or C.

## Models and reliability
- Default: `gemini-3.1-flash-lite` (answers in 1-2 s), fallback `gemini-3.5-flash`.
  In testing (Nov 2026) `gemini-3.5-flash` was often overloaded (timeouts, 503) and
  `gemini-flash-latest` never answered.
- A model that fails (timeout after `LLM_TIMEOUT` = 20 s, 503, 429) is skipped for
  `COOLDOWN_SECONDS` = 120 s, so one overloaded model doesn't slow down every visitor.
- `python check_models.py` shows which models currently answer with your key.

## How the website handles uploads
- Visitors upload their own lecture PDFs in the sidebar. Files are indexed in memory for that
  browser tab only: no one else can search them and nothing is saved on the server.
- With no uploads, the site runs in **demo mode** on the slides in `data/slides/` (index built by
  `ingest.py`), with one-click example questions.
- Limits: 50 MB per file (`.streamlit/config.toml`), 300 pages per visitor
  (`MAX_UPLOAD_PAGES` in `config.py`). Scanned PDFs without a text layer can't be searched.
- Citation tags come from file names: `L03_RAG_basics.pdf` -> `[L03 p.14]`; any other name is
  used as is, e.g. `Week 3.pdf` -> `[Week_3 p.14]`.

## Evaluate
`eval/test_set.jsonl` has 20 questions: 15 answerable (direct, paraphrased, multi-slide,
Korean) and 5 out of scope. Each answerable question lists the slides that contain the answer.
```bash
python evaluate.py --retrieval-only               # search only: no API calls, takes seconds
python evaluate.py --retrieval-only --no-hybrid   # same, meaning-based search only
python evaluate.py --name improved_v1 --judge --delay 3   # full run; the LLM grades answers
```
Options: `--delay N` waits between questions (avoids free-tier 429 errors),
`--min-score 0.8` tries a different refusal threshold, `--no-hybrid` turns keyword search off.
Questions where every model was busy are reported as `n_busy_skipped` and left out of the
percentages, so re-run if that number is high. Results: `eval/summary_*.json`.

| | baseline | improved_v1 |
|---|---|---|
| Search | meaning only | hybrid (meaning + keyword) |
| Model | gemini-flash-latest -> 3.5-flash | 3.1-flash-lite |
| Retrieval hit@5 | 93.3% | 100% |
| Citation accuracy | 93.3% | 100% |
| Correct refusals (out of scope) | 80.0% | 80.0% |
| Answers graded correct | 80.0% | 93.3% (+6.7% partial) |
| Time per answer | ~4 min (main model hung) | ~1-2 s |

The one missed refusal (q19, Mobility Office weekend hours) is a near miss: the slides use the
Mobility Office as an example, and the model replied that the slides don't contain the hours,
just not with the exact refusal sentence the evaluation checks for. q13 ("What must students
submit for the lab?") went from a retrieval miss to a correct answer thanks to keyword search.

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
