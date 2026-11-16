# Lecture Notes Assistant (RAG over course slides)

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # then put your API key in .env
```
Put the lecture PDFs in `data/slides/`, named like `L01_Intro.pdf`, `L03_RAG_basics.pdf`.
The `L01` part becomes the citation tag, e.g. `[L03 p.14]`.

## How the website works
- Visitors upload their own lecture PDFs in the sidebar. Files are indexed in memory for that
  browser tab only: no one else can search them and nothing is saved on the server.
- Every answer shows **📍 Found on** with images of the slides it cites; the other slides
  searched are in a collapsed section.
- With no uploads, the site runs in **demo mode** on the slides in `data/slides/` (index built by
  `ingest.py`).
- Limits: 50 MB per file (`.streamlit/config.toml`), 300 pages per visitor
  (`MAX_UPLOAD_PAGES` in `config.py`). Scanned PDFs without a text layer can't be searched.
- Citation tags come from file names: `L03_RAG_basics.pdf` -> `[L03 p.14]`; any other name is
  used as is, e.g. `Week 3.pdf` -> `[Week_3 p.14]`.

## Run
```bash
python ingest.py                     # build the index (re-run whenever slides change)
python rag.py "What is RAG?"         # quick test in the terminal
streamlit run app.py                 # chat UI at http://localhost:8501
```

## Evaluate
Edit `eval/test_set.jsonl` with your real questions and slide numbers, then:
```bash
python evaluate.py --name baseline
# change something (chunking, k, prompt...), re-run ingest if needed, then:
python evaluate.py --name improved_v1 --judge
```
Options: `--delay 10` waits between questions (avoids free-tier 429 errors),
`--min-score 0.8` tries a different refusal threshold. Questions where every model
was busy are reported as `n_busy_skipped` and left out of the percentages, so
re-run if that number is high.
Compare `eval/summary_*.json` files for the before/after table in the presentation.

## Deploy (public link for Nov 19)
1. Push this folder to a GitHub repo, **including `index/` and `data/slides/`**
   (get the professor's permission to publish the slides first).
2. On share.streamlit.io: New app -> pick the repo -> main file `app.py`.
3. In the app's Settings -> Secrets, add the same three lines as your `.env` (GEMINI_API_KEY, LLM_BASE_URL, LLM_MODEL).
