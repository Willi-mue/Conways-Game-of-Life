import pytest

import hashlife as hl
import i18n


@pytest.fixture(autouse=True)
def restore_language():
    before = i18n.language()
    yield
    i18n.set_language(before)


def test_all_languages_have_the_same_keys():
    keys = [set(texts) for texts in i18n.TEXT.values()]
    assert all(k == keys[0] for k in keys)


def test_every_pattern_has_a_name():
    for lang in i18n.LANGUAGES:
        i18n.set_language(lang)
        for pattern_id in hl.PATTERNS:
            assert i18n.tr(f"pattern.{pattern_id}")


@pytest.mark.parametrize("lang, small, huge", [
    ("de", "1.234.567", "1,23·10^15"),
    ("en", "1,234,567", "1.23·10^15"),
])
def test_number_format(lang, small, huge):
    i18n.set_language(lang)
    assert i18n.fmt_int(1234567) == small
    assert i18n.fmt_int(1_234_567_890_123_456) == huge


def test_command_line_accepts_german_and_english_names():
    import main
    args = main.parse_args(["--theme", "glut", "--view", "kacheln", "--lang", "en"])
    assert (args.theme, args.view) == ("ember", "tiles")
    args = main.parse_args(["--theme", "pastel", "--view", "single", "--lang", "de"])
    assert (args.theme, args.view) == ("pastel", "single")
    assert i18n.language() == "de"
