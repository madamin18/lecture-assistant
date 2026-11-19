"""Interface texts: every English text has a Russian and a Korean one that use the same values,
so switching the language can never break a page with a missing key or a KeyError in
str.format."""
import inspect
import string

import pytest

from ui import i18n

OTHER = [code for code in i18n.TEXT if code != "en"]


def placeholders(text: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_language_has_a_switch_label_and_texts():
    assert set(i18n.LANGUAGES) == set(i18n.TEXT) == {"en", "ru", "ko"}


@pytest.mark.parametrize("code", OTHER)
def test_every_text_is_translated(code):
    assert set(i18n.TEXT[code]) == set(i18n.EN)


@pytest.mark.parametrize("code,key", [(code, key) for code in OTHER for key in sorted(i18n.EN)])
def test_translation_uses_the_same_values(code, key):
    en, other = i18n.EN[key], i18n.TEXT[code][key]
    assert type(en) is type(other) or (callable(en) and callable(other)), key
    if callable(en):
        params = list(inspect.signature(en).parameters)
        assert list(inspect.signature(other).parameters) == params
        values = {name: True if name == "uploaded" else 3 for name in params}
        assert en(**values) and other(**values)
    elif isinstance(en, str):
        assert placeholders(en) == placeholders(other), key
    else:
        assert len(en) == len(other)  # lists, e.g. example questions


def test_russian_plurals():
    forms = ("слайд", "слайда", "слайдов")
    assert [i18n.plural_ru(n, *forms) for n in (1, 2, 4, 5, 11, 12, 21, 22, 25, 101, 111)] == [
        "слайд", "слайда", "слайда", "слайдов", "слайдов", "слайдов", "слайд", "слайда",
        "слайдов", "слайд", "слайдов"]
    assert i18n.RU["badge_slides"](n=491) == "491 слайд"
    assert i18n.RU["badge_models"](n=22, p=3) == "22 модели ИИ от 3 провайдеров"
    assert i18n.KO["badge_models"](n=22, p=3) == "AI 모델 22개 (제공업체 3곳)"


@pytest.mark.parametrize("table", [i18n.REASONS_RU, i18n.REASONS_KO])
def test_failure_reasons_are_translated(table):
    from rag import describe_error

    class E(Exception):
        def __init__(self, status):
            self.status_code = status
    reasons = {describe_error(E(s)) for s in (400, 401, 403, 404, 413, 429, 503)}
    reasons.add(describe_error(Exception()))
    assert reasons <= set(table)
