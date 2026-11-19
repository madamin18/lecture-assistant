"""Browse lectures tab: a card per lecture and a slide viewer, with shortcuts to ask about a
slide or get quizzed on a lecture."""
import streamlit as st

from ui import common as c
from ui import i18n
from ui.i18n import t


def _go_to_page(key: str, pages: list[int], step: int) -> None:
    """Move the viewer by `step` pages, staying inside the valid range."""
    current = st.session_state.get(key, pages[0])
    idx = pages.index(current) if current in pages else 0
    st.session_state[key] = pages[max(0, min(len(pages) - 1, idx + step))]


def _viewer(tag: str):
    title = dict(c.rag().lectures)[tag]
    pdf = c.lecture_file(tag)
    pages = c.page_index(pdf)
    total = c.page_count(pdf)
    st.markdown(f":violet-badge[{tag}] **{title}** · {t('n_slides', n=total)}")

    def slide_name(p: int) -> str:
        label, _, name = pages.get(p, (None, None, ""))
        return f"{label or p} · {c.shorten(name, 40)}" if name else t("slide_option", n=label or p)

    page_values = list(range(1, total + 1))
    key = f"viewer_{tag}"
    current = st.session_state.get(key, page_values[0])
    idx = page_values.index(current) if current in page_values else 0
    prev_col, next_col = st.columns(2)
    prev_col.button(t("prev_slide"), key=f"prev_{tag}", on_click=_go_to_page,
                    args=(key, page_values, -1), disabled=idx == 0)
    next_col.button(t("next_slide"), key=f"next_{tag}", on_click=_go_to_page,
                    args=(key, page_values, 1), disabled=idx == len(page_values) - 1)
    page = st.select_slider(t("slide_label"), options=page_values,
                            key=key, format_func=slide_name)
    label, chunk, name = pages.get(page, (None, None, ""))
    with st.container(horizontal=True):  # above the slide, so no scrolling is needed
        if chunk is not None and st.button(t("ask_slide"), type="primary",
                                           icon=":material/forum:"):
            st.session_state.goto = ("ask", t("ask_slide_q", tag=tag, n=label or page, name=name))
            st.rerun()
        if st.button(t("quiz_lecture"), icon=":material/quiz:"):
            st.session_state.goto = ("quiz", tag)
            st.rerun()
    st.image(c.image(pdf, page), caption=c.slide_caption(title, page, label))


# A dialog's title is fixed when it is defined: one per interface language.
VIEWERS = {"en": st.dialog("Slide viewer", width="large", icon=":material/slideshow:")(_viewer),
           "ru": st.dialog("Просмотр слайдов", width="large", icon=":material/slideshow:")(_viewer),
           "ko": st.dialog("슬라이드 보기", width="large", icon=":material/slideshow:")(_viewer)}


def page():
    st.caption(t("browse_caption"))
    with st.grid("auto", min_column_width=210, gap="small"):
        for tag, title in c.rag().lectures:
            pdf = c.lecture_file(tag)
            with st.container(border=True, height="stretch"):
                st.image(c.image(pdf, 1))
                st.markdown(f":violet-badge[{tag}] **{title}**")
                st.caption(t("n_slides", n=c.page_count(pdf)))
                st.space("stretch")
                if st.button(t("view_slides"), key=f"view_{tag}", icon=":material/slideshow:",
                             width="stretch"):
                    VIEWERS[i18n.lang()](tag)
                with st.container(horizontal=True, gap="small"):
                    st.button(t("ask_btn"), key=f"ask_{tag}", icon=":material/forum:",
                              on_click=c.ask_lecture, args=(tag,), width="stretch")
                    st.button(t("quiz_btn"), key=f"quiz_{tag}", icon=":material/quiz:",
                              on_click=c.quiz_lecture, args=(tag,), width="stretch")
