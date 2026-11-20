"""The mini games' files (ui/game_assets/): word pairs in every interface language, and component
code that uses only the Streamlit custom components v2 API."""
import json
import re
from pathlib import Path

import pytest

from ui import i18n

ASSETS = Path(__file__).resolve().parent.parent / "ui" / "game_assets"
PAIRS = json.loads((ASSETS / "pairs.json").read_text(encoding="utf-8"))


def test_term_match_has_enough_unique_pairs():
    terms = [p["term"] for p in PAIRS]
    assert len(terms) >= 8  # one round uses 8 pairs
    assert len(set(terms)) == len(terms)


def test_every_meaning_is_in_every_interface_language():
    for p in PAIRS:
        for code in i18n.LANGUAGES:
            assert p.get(code, "").strip(), (p["term"], code)
    meanings = [p[code] for p in PAIRS for code in i18n.LANGUAGES]
    assert len(set(meanings)) == len(meanings)  # two cards must never show the same text


@pytest.mark.parametrize("name", ["runner.js", "match.js"])
def test_components_use_the_v2_api_only(name):
    js = (ASSETS / name).read_text(encoding="utf-8")
    assert "export default function" in js
    # v1 globals don't exist in v2 components and would break the game
    for banned in ("Streamlit.", "setComponentValue", "setFrameHeight", "window.parent.postMessage"):
        assert banned not in js, banned
    assert "\n" in js.strip()  # multi-line, so it is read as inline code, not as a file path


@pytest.mark.parametrize("name", ["runner.css", "match.css"])
def test_styles_follow_the_theme(name):
    css = (ASSETS / name).read_text(encoding="utf-8")
    assert "var(--st-" in css
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", css)  # colours come from the theme, not literals
