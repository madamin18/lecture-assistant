"""Interface text in English and Russian. t("key", name=value) gives the text in the language
picked at the top of the sidebar (ENG / RUS). A text is a str.format template, or a function
for words whose form depends on a number (Russian: 1 слайд, 2 слайда, 5 слайдов).

To add a language: add it to LANGUAGES and give it a dictionary in TEXT; a missing key falls
back to English."""
LANGUAGES = {"en": "ENG", "ru": "RUS"}


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
    if lang() == "en":
        return text
    if text.startswith("error "):
        return "ошибка " + text[6:]
    return REASONS_RU.get(text, text)


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

TEXT = {"en": EN, "ru": RU}
