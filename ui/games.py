"""Mini games for a short break ("Bored? Play a game" in the sidebar). Both are original and
use the course's vocabulary:

  Citation Runner  an endless runner: jump over software bugs and traffic cones, run under the
                   hallucination ghosts, collect stars for bonus points
  Term Match       a memory game: flip cards to match each course term with its meaning

Each game is a small custom component (st.components.v2, no iframe, nothing loaded from
outside), coloured with the app's theme variables so it fits light and dark mode. Its CSS and
JavaScript live in ui/game_assets/ (runner.js, match.js, ...) and the Term Match pairs in
ui/game_assets/pairs.json. When a round ends the game reports the result to Python, which keeps
the visitor's best and the best on this server."""
import json
import threading
from pathlib import Path

import streamlit as st

from ui import common as c
from ui import i18n
from ui.i18n import t

ASSETS = Path(__file__).with_name("game_assets")


def _asset(name: str) -> str:
    # Passed as text, not as a path: CCv2 reads multi-line strings as inline code.
    return (ASSETS / name).read_text(encoding="utf-8")


RUNNER_HTML = """<div class="runner"><canvas aria-label="Citation Runner game"></canvas></div>"""
MATCH_HTML = """<div class="match">
  <div class="bar"><span class="moves"></span><span class="time"></span><span class="msg"></span>
    <button type="button" class="new"></button></div>
  <div class="grid"></div>
</div>"""

# Term Match pairs: course terms (kept in English, as on the slides) and their meaning in each
# interface language.
PAIRS = json.loads(_asset("pairs.json"))

_RUNNER = st.components.v2.component("lecture_runner", html=RUNNER_HTML,
                                     css=_asset("runner.css"), js=_asset("runner.js"))
_MATCH = st.components.v2.component("lecture_match", html=MATCH_HTML,
                                    css=_asset("match.css"), js=_asset("match.js"))
GAMES = ["runner", "match"]


@st.cache_resource
def site_records() -> dict:
    """Best results on this server since it started (shared by all visitors)."""
    return {"lock": threading.Lock(), "runner": 0, "match": None}


def _runner_over():
    result = st.session_state["game_runner"].over or {}
    score = int(result.get("score", 0))
    st.session_state.best_runner = max(st.session_state.get("best_runner", 0), score)
    records = site_records()
    with records["lock"]:
        records["runner"] = max(records["runner"], score)


def _match_won():
    result = st.session_state["game_match"].won or {}
    moves = int(result.get("moves", 0))
    if moves:
        mine = st.session_state.get("best_match")
        st.session_state.best_match = moves if mine is None else min(mine, moves)
        records = site_records()
        with records["lock"]:
            records["match"] = moves if records["match"] is None else min(records["match"], moves)
        st.session_state.match_just_won = moves  # shown in the dialog (callbacks can't draw)


def _games():
    st.caption(t("games_caption"))
    game = st.segmented_control(t("games_pick"), GAMES, format_func=lambda g: t(f"game_{g}"),
                                key="game_pick", required=True, label_visibility="collapsed")
    records = site_records()
    if game == "match":
        st.caption(t("match_help"))
        _MATCH(key="game_match", width="stretch", on_won_change=_match_won, data={
            "pairs": [[p["term"], p[i18n.lang()]] for p in PAIRS],
            "text": {"moves": t("match_moves"), "time": t("match_time"),
                     "won": t("match_won"), "new": t("match_new")}})
        won = st.session_state.pop("match_just_won", None)
        if won:
            st.success(t("match_won_toast", moves=won), icon=":material/celebration:")
        mine, site = st.session_state.get("best_match"), records["match"]
        st.caption(t("best_match", mine=mine if mine is not None else "–",
                     site=site if site is not None else "–"))
    else:
        st.caption(t("runner_help"))
        _RUNNER(key="game_runner", width="stretch", on_over_change=_runner_over, data={
            "best": st.session_state.get("best_runner", 0),
            "text": {"start": t("runner_start"), "over": t("runner_over"),
                     "restart": t("runner_restart"), "score": t("runner_score"),
                     "best": t("runner_best"), "jump": t("runner_jump"),
                     "under": t("runner_under"), "bonus": t("runner_bonus")}})
        st.caption(t("best_runner", mine=st.session_state.get("best_runner", 0),
                     site=records["runner"]))


DIALOGS = c.dialog_per_language(_games, "games_title", width="large",
                                icon=":material/sports_esports:")


def open_games():
    st.session_state.setdefault("game_pick", "runner")
    DIALOGS[i18n.lang()]()
