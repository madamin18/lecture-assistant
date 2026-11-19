"""Interface texts: every English text has a Russian one that uses the same values, so switching
the language can never break a page with a missing key or a KeyError in str.format."""
import inspect
import string

import pytest

from ui import i18n


def placeholders(text: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_text_is_translated():
    assert set(i18n.EN) == set(i18n.RU)


@pytest.mark.parametrize("key", sorted(i18n.EN))
def test_translation_uses_the_same_values(key):
    en, ru = i18n.EN[key], i18n.RU[key]
    assert type(en) is type(ru) or (callable(en) and callable(ru)), key
    if callable(en):
        assert list(inspect.signature(en).parameters) == list(inspect.signature(ru).parameters)
        values = {name: 3 for name in inspect.signature(en).parameters}
        if "uploaded" in values:
            values["uploaded"] = True
        assert en(**values) and ru(**values)
    elif isinstance(en, str):
        assert placeholders(en) == placeholders(ru), key
    else:
        assert len(en) == len(ru)  # lists, e.g. example questions


def test_russian_plurals():
    forms = ("слайд", "слайда", "слайдов")
    assert [i18n.plural_ru(n, *forms) for n in (1, 2, 4, 5, 11, 12, 21, 22, 25, 101, 111)] == [
        "слайд", "слайда", "слайда", "слайдов", "слайдов", "слайдов", "слайд", "слайда",
        "слайдов", "слайд", "слайдов"]
    assert i18n.RU["badge_slides"](n=491) == "491 слайд"
    assert i18n.RU["badge_models"](n=22, p=3) == "22 модели ИИ от 3 провайдеров"


def test_failure_reasons_are_translated():
    from rag import describe_error

    class E(Exception):
        def __init__(self, status):
            self.status_code = status
    reasons = {describe_error(E(s)) for s in (400, 401, 403, 404, 413, 429, 503)}
    reasons.add(describe_error(Exception()))
    assert reasons <= set(i18n.REASONS_RU)
