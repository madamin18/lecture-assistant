"""Browse lectures tab: a card per lecture and a slide viewer, with shortcuts to ask about a
slide or get quizzed on a lecture."""
import streamlit as st

from ui import common as c


@st.dialog("Slide viewer", width="large", icon=":material/slideshow:")
def slide_viewer(tag: str):
    title = dict(c.rag().lectures)[tag]
    pdf = c.lecture_file(tag)
    pages = c.page_index(pdf)
    total = c.page_count(pdf)
    st.markdown(f":violet-badge[{tag}] **{title}** · {total} slides")

    def slide_name(p: int) -> str:
        label, _, name = pages.get(p, (None, None, ""))
        return f"{label or p} · {c.shorten(name, 40)}" if name else f"slide {label or p}"

    page = st.select_slider("Slide", options=list(range(1, total + 1)), key=f"viewer_{tag}",
                            format_func=slide_name)
    label, chunk, name = pages.get(page, (None, None, ""))
    with st.container(horizontal=True):  # above the slide, so no scrolling is needed
        if chunk is not None and st.button("Ask about this slide", type="primary",
                                           icon=":material/forum:"):
            st.session_state.goto = ("ask", f'Explain this slide from {tag} (slide {label or page}): '
                                            f'"{name}"')
            st.rerun()
        if st.button("Quiz me on this lecture", icon=":material/quiz:"):
            st.session_state.goto = ("quiz", tag)
            st.rerun()
    st.image(c.image(pdf, page), caption=c.slide_caption(title, page, label))


def page():
    st.caption("Open a lecture to flip through its slides, ask about any slide, or get quizzed.")
    with st.grid("auto", min_column_width=210, gap="small"):
        for tag, title in c.rag().lectures:
            pdf = c.lecture_file(tag)
            with st.container(border=True, height="stretch"):
                st.image(c.image(pdf, 1))
                st.markdown(f":violet-badge[{tag}] **{title}**")
                st.caption(f"{c.page_count(pdf)} slides")
                st.space("stretch")
                if st.button("View slides", key=f"view_{tag}", icon=":material/slideshow:",
                             width="stretch"):
                    slide_viewer(tag)
                with st.container(horizontal=True, gap="small"):
                    st.button("Ask", key=f"ask_{tag}", icon=":material/forum:",
                              on_click=c.ask_lecture, args=(tag,), width="stretch")
                    st.button("Quiz", key=f"quiz_{tag}", icon=":material/quiz:",
                              on_click=c.quiz_lecture, args=(tag,), width="stretch")
