"""Interface text in English, Russian and Korean. t("key", name=value) gives the text in the
language picked at the top of the sidebar (ENG / RUS / KOR). A text is a str.format template, or a function
for words whose form depends on a number (Russian: 1 слайд, 2 слайда, 5 слайдов).

To add a language: add it to LANGUAGES and give it a dictionary in TEXT; a missing key falls
back to English."""
LANGUAGES = {"en": "ENG", "ru": "RUS", "ko": "KOR"}


def lang() -> str:
    import streamlit as st  # here, so the texts can be tested without Streamlit installed
    return st.session_state.get("ui_lang") or "en"


def t(key: str, **kw) -> str:
    text = TEXT[lang()].get(key, TEXT["en"][key])
    if callable(text):
        return text(**kw)
    return text.format(**kw) if kw else text


def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Russian noun form after a number: 1 слайд, 2 слайда, 5 слайдов, 21 слайд."""
    m = abs(n) % 100
    if 11 <= m <= 19:
        return many
    m %= 10
    return one if m == 1 else few if 2 <= m <= 4 else many


def reason(text: str) -> str:
    """A model's failure reason (from rag.describe_error) in the interface language."""
    code = lang()
    if code == "en":
        return text
    if text.startswith("error "):
        return {"ru": "ошибка ", "ko": "오류 "}[code] + text[6:]
    return {"ru": REASONS_RU, "ko": REASONS_KO}[code].get(text, text)


REASONS_RU = {
    "rejected the request": "отклонила запрос",
    "API key rejected": "ключ API отклонён",
    "no access with this key": "нет доступа с этим ключом",
    "model not found": "модель не найдена",
    "request too large for this model's free tier":
        "запрос слишком большой для бесплатного тарифа этой модели",
    "rate limit or daily quota reached": "достигнут лимит запросов или дневная квота",
    "overloaded": "модель перегружена",
    "timed out or couldn't connect": "нет ответа или нет соединения",
    "busy": "занята",
}

EN = {
    # page and header
    "app_title": "Lecture Notes Assistant",
    "app_subtitle": "Ask about your lectures, practise for the exam, or flip through the slides. "
                    "Every answer shows the slide it came from.",
    "ui_language": "Interface language",
    "no_key": "No API key found. Add GEMINI_API_KEY (or GROQ_API_KEY, OPENROUTER_API_KEY) to "
              "`.env` (local) or to the app's Secrets on Streamlit Cloud (Manage app -> Settings "
              "-> Secrets).",
    "badge_lectures": lambda n, uploaded: f"{n} {'uploaded' if uploaded else 'course'} "
                                          f"lecture{'s' * (n != 1)}",
    "badge_slides": lambda n: f"{n} slide{'s' * (n != 1)}",
    "badge_models": lambda n, p: f"{n} AI model{'s' * (n != 1)}"
                                 + (f" from {p} providers" if p > 1 else ""),
    "badge_cites": "Answers cite their slides",
    "demo_caption": "Demo: the lectures of *Design of Generative AI Application Systems*. Upload "
                    "your own PDFs in the sidebar to use those instead.",
    "ask": ":material/forum: Ask",
    "practice": ":material/quiz: Practice exam",
    "browse": ":material/auto_stories: Browse lectures",
    "compare": ":material/compare_arrows: Compare models",

    # uploads
    "upload_bad_pdf": "**{name}** couldn't be opened as a PDF.",
    "upload_too_many": "These files have {pages} pages in total. The limit is {limit}; please "
                       "upload fewer lectures.",
    "upload_reading": "Reading {pages} slides… (first time takes about a minute)",
    "upload_no_text": "**{name}** has no readable text (probably scanned images), so it can't be "
                      "searched.",
    "upload_ready": "Ready: {slides} slides from {files} file(s)",
    "upload_start": "Upload one or more lecture PDFs in the sidebar to get started.",

    # sidebar
    "sb_your_slides": "Your slides",
    "sb_upload": "Upload lecture PDFs",
    "sb_upload_help": "Only you can search the files you upload. They stay in memory for this "
                      "browser tab and are never saved on the server.",
    "sb_filter": "Search only these lectures",
    "sb_filter_help": "Leave empty to search every lecture. Naming a lecture in your question "
                      "(\"in Lesson 3\", \"3강\") also works.",
    "sb_k": "Slides read per answer",
    "sb_k_help": "How many of the best-matching slides the assistant reads before answering.",
    "sb_model": "AI model",
    "auto": "Auto",
    "auto_recommended": "Auto (recommended)",
    "sb_model_help": "Auto starts with the model that did best in our tests (fast and "
                     "accurate) and moves on to the next one, or another provider, when a model "
                     "is busy or out of free quota. Pick a model to try it; the Compare models "
                     "tab puts several side by side.",
    "sb_clear": "Clear chat",
    "sb_save": "Save chat",
    "sb_save_help": "Download this chat as study notes (Markdown), with the slides each answer "
                    "came from.",
    "sb_disclaimer": "Answers and practice questions come only from the slides and can still be "
                     "wrong. Check the slide shown with each one before relying on it.",

    # slides, sources, "How I found this"
    "caption_pdf_page": "{title} · PDF page {page}",
    "caption_slide": "{title} · slide {label}",
    "caption_pdf_suffix": " (PDF page {page})",
    "thanks": "Thanks for the feedback!",
    "searching_only_toast": "Now searching only {tag}. Clear the filter in the sidebar to search "
                            "everything.",
    "api_error": "The AI service rejected the request (error {code}). If you run this site, "
                 "check the API keys in `.env` or the app's Secrets.",
    "busy": "The AI service is busy right now. Please try again in a minute.",
    "notes_title": "# Lecture Notes Assistant: chat",
    "notes_slides": "Slides",
    "scope_lectures": "{n} slides of {lectures}",
    "scope_all": "{n} slides",
    "trace_cached": "Searched {scope} · saved answer, shown instantly",
    "trace_searched": "Searched {scope} in {s:.1f} s",
    "trace_answered": " · answered in {s:.1f} s",
    "step_translated": "Translated the question for searching",
    "scope_named": " (named in the question)",
    "scope_filter": " (sidebar filter)",
    "scope_all_lectures": "all {n} lectures",
    "step_search": "Hybrid search: meaning + keywords, top {k} slides",
    "step_searched_scope": "Searched {scope}.",
    "step_failed": "{model} didn't answer",
    "step_failed_reason": "{reason}, so the next model was asked.",
    "step_answer": "Answer written by {model}",
    "step_answer_detail": "{s:.1f} s, with a citation after each sentence.",
    "step_refused": "No slide was close enough, so the AI wasn't asked",
    "step_refused_detail": "The answer is the standard refusal.",
    "step_cached": "Shown from saved answers",
    "step_cached_detail": "This question was answered before with the same slides and settings, "
                          "so the saved answer is shown instantly instead of asking the AI again.",
    "col_slide": "Slide",
    "col_title": "Title",
    "col_relevance": "Relevance",
    "col_lecture": "Lecture",
    "relevance_help": "Meaning-based similarity between the question and the slide",
    "found_on": "Found on",
    "other_slides": "Other slides searched",
    "slides_searched": "Slides searched",

    # Ask
    "ask_next": "Ask next",
    "explain_from": 'Explain "{title}" from {lec}.',
    "takeaways_label": ":material/summarize: Key takeaways of {tag}",
    "takeaways_q": "What are the key takeaways of {tag}?",
    "welcome_title": ":material/waving_hand: What would you like to know?",
    "welcome_text": "Ask anything about the lectures, in English, Korean or Russian. Every answer "
                    "shows the slides it came from.",
    "welcome_tip": "Tip: name a lecture (\"in Lesson 3\", \"3강\") or use the sidebar filter to "
                   "search only that one.",
    "try_asking": "Try asking",
    "chat_placeholder": "Ask a question about the slides (English, 한국어, Русский)",
    "filter_caption": ":material/filter_alt: Searching only {lectures} (change this in the "
                      "sidebar).",
    "searching": ":shimmer[Searching the slides…]",

    # Practice exam
    "kind_mixed": "Mixed",
    "kind_mcq": "Multiple choice",
    "kind_short": "Short answer",
    "qlang_same": "Same as slides",
    "quiz_spinner": "Writing practice questions from your slides…",
    "quiz_none": "Couldn't write questions from these slides this time. Please try again",
    "quiz_none_topic": " or try a different topic.",
    "quiz_other_model": " A different AI model may do better.",
    "quiz_ready": "{n} questions ready. Good luck!",
    "sheet_title": "# Practice questions",
    "sheet_answers": "## Answers",
    "source": "Source: {cite} · {title}",
    "explain_btn": "Explain this to me",
    "explain_help": "Ask the assistant about this question; it answers from the slides.",
    "q_badge": "Question {n}",
    "your_answer": "Your answer",
    "correct_msg": "Correct: **{letter}) {text}**",
    "wrong_msg": "You picked **{picked}**. The answer is **{letter}) {text}**",
    "short_placeholder": "Write your answer, then reveal the model answer to compare.",
    "reveal": "Reveal model answer",
    "model_answer": "**Model answer:** {text}",
    "did_you": "Did you get it right?",
    "m_answered": "Answered",
    "m_correct": "Correct",
    "m_score": "Score",
    "written_by": ":material/smart_toy: Questions written by {model}",
    "finished_great": "Finished: {c} / {n} ({pct}%). Excellent work!",
    "finished": "Finished: {c} / {n} ({pct}%). Review the source slides for the ones you "
                "missed, then try again.",
    "new_questions": "New questions",
    "retry_missed": "Retry the {n} I missed",
    "try_again": "Try again",
    "download_sheet": "Download study sheet",
    "build_exam": ":material/edit_note: Build a practice exam",
    "lectures_pick": "Lectures (leave empty for all)",
    "topic": "Topic (optional)",
    "topic_placeholder": "e.g. hallucinations. Leave empty to cover everything.",
    "n_questions": "Questions",
    "q_type": "Type",
    "q_lang": "Language",
    "generate": "Generate questions",
    "practice_caption": ":material/info: Questions are checked against the slides they cite. "
                        "Answer one and you'll see right away whether you got it.",

    # Browse lectures
    "browse_caption": "Open a lecture to flip through its slides, ask about any slide, or get "
                      "quizzed.",
    "view_slides": "View slides",
    "ask_btn": "Ask",
    "quiz_btn": "Quiz",
    "n_slides": lambda n: f"{n} slide{'s' * (n != 1)}",
    "slide_label": "Slide",
    "slide_option": "slide {n}",
    "ask_slide": "Ask about this slide",
    "ask_slide_q": 'Explain this slide from {tag} (slide {n}): "{name}"',
    "quiz_lecture": "Quiz me on this lecture",

    # Compare models
    "cmp_title": ":material/compare_arrows: Which AI model answers best?",
    "cmp_caption": "Every model gets the same question and the same slides, so only the writing "
                   "differs. Vote for the better answer; the names stay hidden until you vote.",
    "cmp_one_model": "Only one AI model is available. Add another provider's API key (e.g. "
                     "GROQ_API_KEY) to compare models.",
    "cmp_models": "Models (2 or 3)",
    "cmp_question": "Question",
    "cmp_placeholder": "Ask about the lectures, in English, Korean or Russian",
    "cmp_blind": "Hide model names until I vote",
    "cmp_go": "Compare answers",
    "cmp_pick2": "Pick at least two models.",
    "cmp_need_q": "Type a question first.",
    "cmp_searching": "Searching the slides…",
    "cmp_writing": "{name} is writing…",
    "cmp_model_x": "Model {letter}",
    "cmp_your_pick": ":green-badge[:material/thumb_up: your pick]",
    "cmp_no_answer": "No answer: {reason}.",
    "cmp_secs": "{s:.1f} s",
    "cmp_cited": lambda n: f"{n} slide{'s' * (n != 1)} cited",
    "cmp_made_up": lambda n: f"{n} made-up citation{'s' * (n != 1)}",
    "cmp_refused": "refused",
    "cmp_vote_q": "**Which answer is better?** Check the cited slides below before you vote.",
    "cmp_better": "{name} is better",
    "cmp_tie": "Tie",
    "cmp_both_bad": "Both bad",
    "cmp_all_bad": "All bad",
    "cmp_voted": "You voted for {verdict}.",
    "cmp_verdict_tie": "a tie",
    "cmp_verdict_bad": "no good answer",
    "cmp_thanks": "Thanks! Your vote is in the leaderboard below.",
    "cmp_refusal_all": "No slide is close enough to this question, so every model would get the "
                       "same refusal. Try a question about the lectures.",
    "cmp_slides_read": "Slides every model read ({n})",
    "cmp_votes_title": "##### :material/leaderboard: Visitors' votes",
    "cmp_no_votes": "No votes yet. Compare two models above and vote for the better answer.",
    "col_model": "Model",
    "col_win_rate": "Win rate",
    "col_votes": "Votes",
    "col_ties": "Ties",
    "col_both_bad": "Both bad",
    "col_avg_time": "Avg time (s)",
    "col_no_answer": "No answer",
    "win_rate_help": "Wins, plus half of each tie, out of the comparisons it answered",
    "cmp_votes_caption": "Votes from everyone using this site since the server last restarted. "
                         "A tie counts as half a win.",
    "bench_title": "##### :material/science: Measured on the test set",
    "col_correct": "Correct",
    "col_partial": "Partly correct",
    "col_right_slide": "Cites a right slide",
    "col_made_up": "Made-up citations",
    "col_false_refusal": "Wrongly refused",
    "col_off_topic": "Refused off-topic",
    "bench_caption": "{n} questions from eval/test_set.jsonl ({date}); percentages. Answers "
                     "graded by {judge}, the same judge for every model.",
    "cmp_examples": ["What is a hallucination?", "How is recall@k calculated?",
                     "Why does chunk overlap help retrieval?",
                     "에이전트와 워크플로의 차이는 무엇인가요?"],
}

RU = {
    # page and header
    "app_title": "Помощник по лекциям",
    "app_subtitle": "Задавайте вопросы по лекциям, готовьтесь к экзамену или листайте слайды. "
                    "Каждый ответ показывает слайд, на котором он основан.",
    "ui_language": "Язык интерфейса",
    "no_key": "Ключ API не найден. Добавьте GEMINI_API_KEY (или GROQ_API_KEY, "
              "OPENROUTER_API_KEY) в `.env` (локально) или в Secrets приложения на Streamlit Cloud "
              "(Manage app -> Settings -> Secrets).",
    "badge_lectures": lambda n, uploaded: f"{n} " + (
        plural_ru(n, "загруженная лекция", "загруженные лекции", "загруженных лекций")
        if uploaded else plural_ru(n, "лекция курса", "лекции курса", "лекций курса")),
    "badge_slides": lambda n: f"{n} {plural_ru(n, 'слайд', 'слайда', 'слайдов')}",
    "badge_models": lambda n, p: f"{n} {plural_ru(n, 'модель', 'модели', 'моделей')} ИИ"
                                 + (f" от {p} {plural_ru(p, 'провайдера', 'провайдеров', 'провайдеров')}"
                                    if p > 1 else ""),
    "badge_cites": "Ответы ссылаются на слайды",
    "demo_caption": "Демо: лекции курса *Design of Generative AI Application Systems*. Загрузите "
                    "свои PDF на боковой панели, чтобы работать с ними.",
    "ask": ":material/forum: Вопросы",
    "practice": ":material/quiz: Пробный экзамен",
    "browse": ":material/auto_stories: Слайды лекций",
    "compare": ":material/compare_arrows: Сравнить модели",

    # uploads
    "upload_bad_pdf": "Не удалось открыть **{name}** как PDF.",
    "upload_too_many": "В этих файлах всего {pages} стр. Ограничение — {limit}; загрузите "
                       "меньше лекций.",
    "upload_reading": "Читаю слайды ({pages})… (в первый раз это занимает около минуты)",
    "upload_no_text": "В **{name}** нет распознаваемого текста (вероятно, это сканы), поэтому по "
                      "нему нельзя искать.",
    "upload_ready": "Готово: слайдов — {slides}, файлов — {files}",
    "upload_start": "Чтобы начать, загрузите один или несколько PDF с лекциями на боковой "
                    "панели.",

    # sidebar
    "sb_your_slides": "Ваши слайды",
    "sb_upload": "Загрузить PDF лекций",
    "sb_upload_help": "Искать по загруженным файлам можете только вы. Они хранятся в памяти "
                      "этой вкладки браузера и никогда не сохраняются на сервере.",
    "sb_filter": "Искать только в этих лекциях",
    "sb_filter_help": "Оставьте пустым, чтобы искать во всех лекциях. Можно также назвать "
                      "лекцию в вопросе («в лекции 3», «Lesson 3»).",
    "sb_k": "Слайдов на один ответ",
    "sb_k_help": "Сколько самых подходящих слайдов помощник читает перед ответом.",
    "sb_model": "Модель ИИ",
    "auto": "Авто",
    "auto_recommended": "Авто (рекомендуется)",
    "sb_model_help": "«Авто» начинает с модели, которая лучше всего показала себя в наших тестах "
                     "(быстрой и точной), и переходит к следующей (или к другому провайдеру), "
                     "если модель занята или закончилась бесплатная квота. Выберите модель, "
                     "чтобы попробовать её; вкладка «Сравнить модели» показывает несколько "
                     "моделей рядом.",
    "sb_clear": "Очистить чат",
    "sb_save": "Сохранить чат",
    "sb_save_help": "Скачать чат как конспект (Markdown) со слайдами, на которых основан каждый "
                    "ответ.",
    "sb_disclaimer": "Ответы и вопросы для практики берутся только из слайдов, но всё равно "
                     "могут быть неточными. Прежде чем полагаться на ответ, проверьте "
                     "показанный слайд.",

    # slides, sources, "How I found this"
    "caption_pdf_page": "{title} · стр. PDF {page}",
    "caption_slide": "{title} · слайд {label}",
    "caption_pdf_suffix": " (стр. PDF {page})",
    "thanks": "Спасибо за отзыв!",
    "searching_only_toast": "Теперь поиск только в {tag}. Очистите фильтр на боковой панели, "
                            "чтобы искать везде.",
    "api_error": "Сервис ИИ отклонил запрос (ошибка {code}). Если это ваш сайт, проверьте "
                 "ключи API в `.env` или в Secrets приложения.",
    "busy": "Сервис ИИ сейчас занят. Попробуйте ещё раз через минуту.",
    "notes_title": "# Помощник по лекциям: чат",
    "notes_slides": "Слайды",
    "scope_lectures": "слайдов в {lectures}: {n}",
    "scope_all": "слайдов: {n}",
    "trace_cached": "Просмотрено {scope} · сохранённый ответ, показан сразу",
    "trace_searched": "Просмотрено {scope} за {s:.1f} с",
    "trace_answered": " · ответ за {s:.1f} с",
    "step_translated": "Вопрос переведён для поиска",
    "scope_named": " (названа в вопросе)",
    "scope_filter": " (фильтр на боковой панели)",
    "scope_all_lectures": "все лекции ({n})",
    "step_search": "Гибридный поиск (смысл + ключевые слова), топ-{k} слайдов",
    "step_searched_scope": "Где искали: {scope}.",
    "step_failed": "{model} не ответила",
    "step_failed_reason": "Причина: {reason}, поэтому спросили следующую модель.",
    "step_answer": "Ответ написала модель {model}",
    "step_answer_detail": "{s:.1f} с, со ссылкой после каждого предложения.",
    "step_refused": "Ни один слайд не подошёл, поэтому ИИ не спрашивали",
    "step_refused_detail": "Ответ — стандартный отказ.",
    "step_cached": "Показан сохранённый ответ",
    "step_cached_detail": "На этот вопрос уже отвечали с теми же слайдами и настройками, "
                          "поэтому сохранённый ответ показан сразу, без нового запроса к ИИ.",
    "col_slide": "Слайд",
    "col_title": "Заголовок",
    "col_relevance": "Релевантность",
    "col_lecture": "Лекция",
    "relevance_help": "Смысловое сходство вопроса и слайда",
    "found_on": "Найдено на",
    "other_slides": "Другие просмотренные слайды",
    "slides_searched": "Просмотренные слайды",

    # Ask
    "ask_next": "Спросить дальше",
    "explain_from": "Объясни «{title}» из {lec}.",
    "takeaways_label": ":material/summarize: Главное в {tag}",
    "takeaways_q": "Каковы главные выводы {tag}?",
    "welcome_title": ":material/waving_hand: Что вы хотите узнать?",
    "welcome_text": "Спрашивайте что угодно о лекциях — на русском, английском или корейском. "
                    "Каждый ответ показывает слайды, на которых он основан.",
    "welcome_tip": "Совет: назовите лекцию («в лекции 3», «Lesson 3») или используйте фильтр "
                   "на боковой панели, чтобы искать только в ней.",
    "try_asking": "Попробуйте спросить",
    "chat_placeholder": "Задайте вопрос по слайдам (русский, English, 한국어)",
    "filter_caption": ":material/filter_alt: Поиск только в {lectures} (можно изменить на "
                      "боковой панели).",
    "searching": ":shimmer[Ищу в слайдах…]",

    # Practice exam
    "kind_mixed": "Смешанные",
    "kind_mcq": "С выбором ответа",
    "kind_short": "Краткий ответ",
    "qlang_same": "Как в слайдах",
    "quiz_spinner": "Составляю вопросы по вашим слайдам…",
    "quiz_none": "На этот раз не удалось составить вопросы по этим слайдам. Попробуйте ещё раз",
    "quiz_none_topic": " или выберите другую тему.",
    "quiz_other_model": " Другая модель ИИ может справиться лучше.",
    "quiz_ready": "Вопросы готовы ({n}). Удачи!",
    "sheet_title": "# Вопросы для практики",
    "sheet_answers": "## Ответы",
    "source": "Источник: {cite} · {title}",
    "explain_btn": "Объясни мне это",
    "explain_help": "Спросить помощника об этом вопросе; он ответит по слайдам.",
    "q_badge": "Вопрос {n}",
    "your_answer": "Ваш ответ",
    "correct_msg": "Верно: **{letter}) {text}**",
    "wrong_msg": "Вы выбрали **{picked}**. Правильный ответ: **{letter}) {text}**",
    "short_placeholder": "Напишите ответ, затем откройте образцовый ответ для сравнения.",
    "reveal": "Показать образцовый ответ",
    "model_answer": "**Образцовый ответ:** {text}",
    "did_you": "Вы ответили правильно?",
    "m_answered": "Отвечено",
    "m_correct": "Верно",
    "m_score": "Результат",
    "written_by": ":material/smart_toy: Вопросы составила модель {model}",
    "finished_great": "Готово: {c} / {n} ({pct}%). Отличная работа!",
    "finished": "Готово: {c} / {n} ({pct}%). Посмотрите исходные слайды для вопросов с "
                "ошибками и попробуйте снова.",
    "new_questions": "Новые вопросы",
    "retry_missed": "Повторить ошибки ({n})",
    "try_again": "Пройти заново",
    "download_sheet": "Скачать лист с вопросами",
    "build_exam": ":material/edit_note: Составить пробный экзамен",
    "lectures_pick": "Лекции (пусто — все)",
    "topic": "Тема (необязательно)",
    "topic_placeholder": "например, галлюцинации. Оставьте пустым, чтобы охватить всё.",
    "n_questions": "Вопросов",
    "q_type": "Тип",
    "q_lang": "Язык",
    "generate": "Создать вопросы",
    "practice_caption": ":material/info: Каждый вопрос проверяется по слайду, на который он "
                        "ссылается. Ответьте — и сразу увидите, правильно ли.",

    # Browse lectures
    "browse_caption": "Откройте лекцию, чтобы листать слайды, спросить о любом слайде или "
                      "пройти тест.",
    "view_slides": "Смотреть слайды",
    "ask_btn": "Спросить",
    "quiz_btn": "Тест",
    "n_slides": lambda n: f"{n} {plural_ru(n, 'слайд', 'слайда', 'слайдов')}",
    "slide_label": "Слайд",
    "slide_option": "слайд {n}",
    "ask_slide": "Спросить об этом слайде",
    "ask_slide_q": "Объясни этот слайд из {tag} (слайд {n}): «{name}»",
    "quiz_lecture": "Тест по этой лекции",

    # Compare models
    "cmp_title": ":material/compare_arrows: Какая модель ИИ отвечает лучше?",
    "cmp_caption": "Каждая модель получает один и тот же вопрос и одни и те же слайды, поэтому "
                   "различается только сам ответ. Проголосуйте за лучший ответ; названия "
                   "моделей скрыты до голосования.",
    "cmp_one_model": "Доступна только одна модель ИИ. Добавьте ключ API другого провайдера "
                     "(например, GROQ_API_KEY), чтобы сравнивать модели.",
    "cmp_models": "Модели (2 или 3)",
    "cmp_question": "Вопрос",
    "cmp_placeholder": "Вопрос по лекциям (на русском, английском или корейском)",
    "cmp_blind": "Скрывать названия моделей до голосования",
    "cmp_go": "Сравнить ответы",
    "cmp_pick2": "Выберите хотя бы две модели.",
    "cmp_need_q": "Сначала введите вопрос.",
    "cmp_searching": "Ищу в слайдах…",
    "cmp_writing": "{name} пишет ответ…",
    "cmp_model_x": "Модель {letter}",
    "cmp_your_pick": ":green-badge[:material/thumb_up: ваш выбор]",
    "cmp_no_answer": "Нет ответа: {reason}.",
    "cmp_secs": "{s:.1f} с",
    "cmp_cited": lambda n: f"ссылок на слайды: {n}",
    "cmp_made_up": lambda n: f"выдуманных ссылок: {n}",
    "cmp_refused": "отказ",
    "cmp_vote_q": "**Какой ответ лучше?** Перед голосованием проверьте слайды ниже.",
    "cmp_better": "{name} лучше",
    "cmp_tie": "Ничья",
    "cmp_both_bad": "Оба плохие",
    "cmp_all_bad": "Все плохие",
    "cmp_voted": "Ваш голос: {verdict}.",
    "cmp_verdict_tie": "ничья",
    "cmp_verdict_bad": "хорошего ответа нет",
    "cmp_thanks": "Спасибо! Ваш голос учтён в рейтинге ниже.",
    "cmp_refusal_all": "Ни один слайд не подходит к этому вопросу, поэтому все модели дали бы "
                       "одинаковый отказ. Задайте вопрос по лекциям.",
    "cmp_slides_read": "Слайды, которые прочитала каждая модель ({n})",
    "cmp_votes_title": "##### :material/leaderboard: Голоса посетителей",
    "cmp_no_votes": "Голосов пока нет. Сравните две модели выше и проголосуйте за лучший ответ.",
    "col_model": "Модель",
    "col_win_rate": "Доля побед",
    "col_votes": "Голоса",
    "col_ties": "Ничьи",
    "col_both_bad": "Оба плохие",
    "col_avg_time": "Ср. время (с)",
    "col_no_answer": "Без ответа",
    "win_rate_help": "Победы плюс половина ничьих, от числа сравнений с ответом",
    "cmp_votes_caption": "Голоса всех посетителей сайта с последнего перезапуска сервера. "
                         "Ничья считается за половину победы.",
    "bench_title": "##### :material/science: Измерено на тестовом наборе",
    "col_correct": "Верно",
    "col_partial": "Частично верно",
    "col_right_slide": "Ссылка на верный слайд",
    "col_made_up": "Выдуманные ссылки",
    "col_false_refusal": "Ошибочные отказы",
    "col_off_topic": "Отказы не по теме",
    "bench_caption": "{n} вопросов из eval/test_set.jsonl ({date}); в процентах. Ответы "
                     "оценивала модель {judge}, одна и та же для всех моделей.",
    "cmp_examples": ["Что такое галлюцинация?", "Как вычисляется recall@k?",
                     "Почему перекрытие чанков помогает поиску?",
                     "What is the difference between a workflow and an agent?"],
}

REASONS_KO = {
    "rejected the request": "요청을 거부함",
    "API key rejected": "API 키가 거부됨",
    "no access with this key": "이 키로는 접근 권한이 없음",
    "model not found": "모델을 찾을 수 없음",
    "request too large for this model's free tier": "이 모델의 무료 등급에는 요청이 너무 큼",
    "rate limit or daily quota reached": "요청 한도 또는 일일 할당량에 도달함",
    "overloaded": "모델 과부하",
    "timed out or couldn't connect": "시간 초과 또는 연결 실패",
    "busy": "사용 중",
}

KO = {
    # page and header
    "app_title": "강의 노트 도우미",
    "app_subtitle": "강의 내용을 질문하고, 시험을 준비하고, 슬라이드를 넘겨 보세요. 모든 답변에는 "
                    "근거가 된 슬라이드가 함께 표시됩니다.",
    "ui_language": "인터페이스 언어",
    "no_key": "API 키를 찾을 수 없습니다. GEMINI_API_KEY(또는 GROQ_API_KEY, OPENROUTER_API_KEY)를 "
              "`.env`(로컬) 또는 Streamlit Cloud 앱의 Secrets(Manage app -> Settings -> Secrets)에 "
              "추가하세요.",
    "badge_lectures": lambda n, uploaded: f"{'업로드한 강의' if uploaded else '강의'} {n}개",
    "badge_slides": lambda n: f"슬라이드 {n}장",
    "badge_models": lambda n, p: f"AI 모델 {n}개" + (f" (제공업체 {p}곳)" if p > 1 else ""),
    "badge_cites": "답변마다 출처 슬라이드 표시",
    "demo_caption": "데모: *Design of Generative AI Application Systems* 강의 자료입니다. 사이드바에 "
                    "직접 PDF를 업로드하면 그 자료로 사용할 수 있습니다.",
    "ask": ":material/forum: 질문하기",
    "practice": ":material/quiz: 모의고사",
    "browse": ":material/auto_stories: 강의 둘러보기",
    "compare": ":material/compare_arrows: 모델 비교",

    # uploads
    "upload_bad_pdf": "**{name}** 파일을 PDF로 열 수 없습니다.",
    "upload_too_many": "이 파일들은 모두 {pages}쪽입니다. 한도는 {limit}쪽이니 강의 수를 줄여서 "
                       "업로드해 주세요.",
    "upload_reading": "슬라이드 {pages}장을 읽는 중… (처음에는 1분 정도 걸립니다)",
    "upload_no_text": "**{name}**에는 읽을 수 있는 텍스트가 없어(스캔 이미지로 보임) 검색할 수 "
                      "없습니다.",
    "upload_ready": "준비 완료: 파일 {files}개에서 슬라이드 {slides}장",
    "upload_start": "시작하려면 사이드바에서 강의 PDF를 하나 이상 업로드하세요.",

    # sidebar
    "sb_your_slides": "내 슬라이드",
    "sb_upload": "강의 PDF 업로드",
    "sb_upload_help": "업로드한 파일은 본인만 검색할 수 있습니다. 이 브라우저 탭의 메모리에만 있고 "
                      "서버에는 저장되지 않습니다.",
    "sb_filter": "이 강의에서만 검색",
    "sb_filter_help": "비워 두면 모든 강의를 검색합니다. 질문에 강의를 적어도 됩니다(\"3강\", "
                      "\"Lesson 3\").",
    "sb_k": "답변당 읽는 슬라이드 수",
    "sb_k_help": "답하기 전에 가장 관련 있는 슬라이드를 몇 장 읽을지 정합니다.",
    "sb_model": "AI 모델",
    "auto": "자동",
    "auto_recommended": "자동 (권장)",
    "sb_model_help": "자동은 테스트에서 가장 좋은 결과를 낸 모델(빠르고 정확함)부터 쓰고, 모델이 "
                     "바쁘거나 무료 할당량을 다 쓰면 다음 모델이나 다른 제공업체로 넘어갑니다. "
                     "모델을 직접 골라 써 볼 수도 있고, '모델 비교' 탭에서 여러 모델을 나란히 볼 수 "
                     "있습니다.",
    "sb_clear": "대화 지우기",
    "sb_save": "대화 저장",
    "sb_save_help": "이 대화를 각 답변의 출처 슬라이드와 함께 학습 노트(Markdown)로 내려받습니다.",
    "sb_disclaimer": "답변과 연습 문제는 슬라이드에서만 만들어지지만 틀릴 수 있습니다. 믿기 전에 "
                     "함께 표시된 슬라이드를 확인하세요.",

    # slides, sources, "How I found this"
    "caption_pdf_page": "{title} · PDF {page}쪽",
    "caption_slide": "{title} · 슬라이드 {label}",
    "caption_pdf_suffix": " (PDF {page}쪽)",
    "thanks": "피드백 감사합니다!",
    "searching_only_toast": "이제 {tag}에서만 검색합니다. 전체를 검색하려면 사이드바의 필터를 "
                            "지우세요.",
    "api_error": "AI 서비스가 요청을 거부했습니다(오류 {code}). 이 사이트를 운영한다면 `.env` 또는 "
                 "앱 Secrets의 API 키를 확인하세요.",
    "busy": "AI 서비스가 지금 바쁩니다. 1분 후에 다시 시도해 주세요.",
    "notes_title": "# 강의 노트 도우미: 대화",
    "notes_slides": "슬라이드",
    "scope_lectures": "{lectures}의 슬라이드 {n}장",
    "scope_all": "슬라이드 {n}장",
    "trace_cached": "{scope} 검색 · 저장된 답변을 바로 표시",
    "trace_searched": "{scope}을 {s:.1f}초 만에 검색",
    "trace_answered": " · {s:.1f}초 만에 답변",
    "step_translated": "검색을 위해 질문을 번역함",
    "scope_named": " (질문에 언급됨)",
    "scope_filter": " (사이드바 필터)",
    "scope_all_lectures": "전체 강의 {n}개",
    "step_search": "하이브리드 검색: 의미 + 키워드, 상위 슬라이드 {k}장",
    "step_searched_scope": "검색 범위: {scope}.",
    "step_failed": "{model} 응답 없음",
    "step_failed_reason": "원인: {reason}. 그래서 다음 모델에 물었습니다.",
    "step_answer": "답변 작성: {model}",
    "step_answer_detail": "{s:.1f}초, 문장마다 출처 표시.",
    "step_refused": "충분히 가까운 슬라이드가 없어 AI에 묻지 않음",
    "step_refused_detail": "답변은 표준 거절 문구입니다.",
    "step_cached": "저장된 답변에서 표시",
    "step_cached_detail": "같은 슬라이드와 설정으로 이미 답한 질문이라, AI에 다시 묻지 않고 저장된 "
                          "답변을 바로 보여 줍니다.",
    "col_slide": "슬라이드",
    "col_title": "제목",
    "col_relevance": "관련도",
    "col_lecture": "강의",
    "relevance_help": "질문과 슬라이드의 의미 유사도",
    "found_on": "출처",
    "other_slides": "검색된 다른 슬라이드",
    "slides_searched": "검색된 슬라이드",

    # Ask
    "ask_next": "이어서 물어보기",
    "explain_from": '{lec}의 "{title}"을(를) 설명해 주세요.',
    "takeaways_label": ":material/summarize: {tag} 핵심 정리",
    "takeaways_q": "{tag}의 핵심 내용은 무엇인가요?",
    "welcome_title": ":material/waving_hand: 무엇이 궁금하세요?",
    "welcome_text": "강의에 대해 무엇이든 한국어, 영어, 러시아어로 물어보세요. 모든 답변에 근거 "
                    "슬라이드가 표시됩니다.",
    "welcome_tip": "팁: 질문에 강의를 적거나(\"3강\", \"Lesson 3\") 사이드바 필터를 쓰면 그 강의에서만 "
                   "검색합니다.",
    "try_asking": "이렇게 물어보세요",
    "chat_placeholder": "슬라이드에 대해 질문하세요 (한국어, English, Русский)",
    "filter_caption": ":material/filter_alt: {lectures}에서만 검색 중 (사이드바에서 바꿀 수 "
                      "있습니다).",
    "searching": ":shimmer[슬라이드 검색 중…]",

    # Practice exam
    "kind_mixed": "혼합",
    "kind_mcq": "객관식",
    "kind_short": "단답형",
    "qlang_same": "슬라이드와 같은 언어",
    "quiz_spinner": "슬라이드로 연습 문제를 만드는 중…",
    "quiz_none": "이번에는 이 슬라이드로 문제를 만들지 못했습니다. 다시 시도해 주세요",
    "quiz_none_topic": " 또는 다른 주제를 입력해 보세요.",
    "quiz_other_model": " 다른 AI 모델이 더 잘할 수도 있습니다.",
    "quiz_ready": "문제 {n}개 준비 완료. 행운을 빕니다!",
    "sheet_title": "# 연습 문제",
    "sheet_answers": "## 정답",
    "source": "출처: {cite} · {title}",
    "explain_btn": "이 문제 설명해 줘",
    "explain_help": "이 문제에 대해 도우미에게 물어봅니다. 슬라이드를 근거로 답합니다.",
    "q_badge": "문제 {n}",
    "your_answer": "내 답",
    "correct_msg": "정답: **{letter}) {text}**",
    "wrong_msg": "선택한 답: **{picked}** · 정답: **{letter}) {text}**",
    "short_placeholder": "답을 쓴 뒤 모범 답안을 열어 비교해 보세요.",
    "reveal": "모범 답안 보기",
    "model_answer": "**모범 답안:** {text}",
    "did_you": "맞혔나요?",
    "m_answered": "푼 문제",
    "m_correct": "정답",
    "m_score": "점수",
    "written_by": ":material/smart_toy: 문제 작성: {model}",
    "finished_great": "완료: {c} / {n} ({pct}%). 아주 잘했어요!",
    "finished": "완료: {c} / {n} ({pct}%). 틀린 문제의 출처 슬라이드를 복습한 뒤 다시 풀어 보세요.",
    "new_questions": "새 문제",
    "retry_missed": "틀린 {n}문제 다시 풀기",
    "try_again": "처음부터 다시",
    "download_sheet": "문제지 내려받기",
    "build_exam": ":material/edit_note: 모의고사 만들기",
    "lectures_pick": "강의 (비우면 전체)",
    "topic": "주제 (선택)",
    "topic_placeholder": "예: 환각. 비워 두면 전체 범위에서 출제합니다.",
    "n_questions": "문항 수",
    "q_type": "유형",
    "q_lang": "언어",
    "generate": "문제 만들기",
    "practice_caption": ":material/info: 모든 문제는 인용한 슬라이드와 대조해 검증됩니다. 답을 "
                        "고르면 바로 정답 여부를 알 수 있습니다.",

    # Browse lectures
    "browse_caption": "강의를 열어 슬라이드를 넘겨 보고, 원하는 슬라이드에 대해 묻거나 퀴즈를 풀어 "
                      "보세요.",
    "view_slides": "슬라이드 보기",
    "ask_btn": "질문",
    "quiz_btn": "퀴즈",
    "n_slides": lambda n: f"슬라이드 {n}장",
    "slide_label": "슬라이드",
    "slide_option": "슬라이드 {n}",
    "ask_slide": "이 슬라이드에 대해 묻기",
    "ask_slide_q": '{tag}의 이 슬라이드(슬라이드 {n})를 설명해 주세요: "{name}"',
    "quiz_lecture": "이 강의로 퀴즈 풀기",

    # Compare models
    "cmp_title": ":material/compare_arrows: 어떤 AI 모델이 가장 잘 답할까요?",
    "cmp_caption": "모든 모델이 같은 질문과 같은 슬라이드를 받으므로 답변을 쓰는 방식만 다릅니다. "
                   "더 나은 답변에 투표하세요. 투표하기 전까지 모델 이름은 숨겨집니다.",
    "cmp_one_model": "사용할 수 있는 AI 모델이 하나뿐입니다. 모델을 비교하려면 다른 제공업체의 API "
                     "키(예: GROQ_API_KEY)를 추가하세요.",
    "cmp_models": "모델 (2개 또는 3개)",
    "cmp_question": "질문",
    "cmp_placeholder": "강의에 대해 질문하세요 (한국어, 영어, 러시아어)",
    "cmp_blind": "투표할 때까지 모델 이름 숨기기",
    "cmp_go": "답변 비교하기",
    "cmp_pick2": "모델을 두 개 이상 고르세요.",
    "cmp_need_q": "먼저 질문을 입력하세요.",
    "cmp_searching": "슬라이드 검색 중…",
    "cmp_writing": "{name} 답변 작성 중…",
    "cmp_model_x": "모델 {letter}",
    "cmp_your_pick": ":green-badge[:material/thumb_up: 내 선택]",
    "cmp_no_answer": "답변 없음: {reason}.",
    "cmp_secs": "{s:.1f}초",
    "cmp_cited": lambda n: f"인용 슬라이드 {n}장",
    "cmp_made_up": lambda n: f"지어낸 인용 {n}개",
    "cmp_refused": "거절",
    "cmp_vote_q": "**어느 답변이 더 낫나요?** 투표하기 전에 아래 인용 슬라이드를 확인하세요.",
    "cmp_better": "{name} 선택",
    "cmp_tie": "무승부",
    "cmp_both_bad": "둘 다 별로",
    "cmp_all_bad": "모두 별로",
    "cmp_voted": "내 투표: {verdict}.",
    "cmp_verdict_tie": "무승부",
    "cmp_verdict_bad": "좋은 답변 없음",
    "cmp_thanks": "감사합니다! 아래 순위표에 투표가 반영되었습니다.",
    "cmp_refusal_all": "이 질문과 충분히 가까운 슬라이드가 없어 모든 모델이 같은 거절 답변을 하게 "
                       "됩니다. 강의에 관한 질문을 해 보세요.",
    "cmp_slides_read": "모든 모델이 읽은 슬라이드 ({n}장)",
    "cmp_votes_title": "##### :material/leaderboard: 방문자 투표",
    "cmp_no_votes": "아직 투표가 없습니다. 위에서 두 모델을 비교하고 더 나은 답변에 투표하세요.",
    "col_model": "모델",
    "col_win_rate": "승률",
    "col_votes": "투표 수",
    "col_ties": "무승부",
    "col_both_bad": "둘 다 별로",
    "col_avg_time": "평균 시간(초)",
    "col_no_answer": "답변 없음",
    "win_rate_help": "답변한 비교 중 이긴 비율 (무승부는 0.5승)",
    "cmp_votes_caption": "서버가 마지막으로 다시 시작된 뒤 이 사이트 방문자 모두의 투표입니다. "
                         "무승부는 0.5승으로 계산합니다.",
    "bench_title": "##### :material/science: 테스트 세트로 측정한 결과",
    "col_correct": "정답",
    "col_partial": "부분 정답",
    "col_right_slide": "올바른 슬라이드 인용",
    "col_made_up": "지어낸 인용",
    "col_false_refusal": "잘못된 거절",
    "col_off_topic": "주제 밖 질문 거절",
    "bench_caption": "eval/test_set.jsonl의 질문 {n}개 ({date}), 단위 %. 모든 모델의 답변을 같은 "
                     "채점 모델 {judge}가 평가했습니다.",
    "cmp_examples": ["환각이란 무엇인가요?", "recall@k는 어떻게 계산하나요?",
                     "청크 오버랩은 왜 검색에 도움이 되나요?",
                     "What is the difference between a workflow and an agent?"],
}

TEXT = {"en": EN, "ru": RU, "ko": KO}
