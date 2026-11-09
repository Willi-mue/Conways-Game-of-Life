"""User-interface texts in German and English.

``tr("key", **values)`` returns the text in the current language. The language is chosen from the
system locale at start-up and can be switched at runtime (menu) or with ``--lang``.
"""
import locale

LANGUAGES = ("de", "en")

TEXT = {
    "de": {
        "menu.title": "Einstellungen",
        "menu.open": "Menü (M)",
        "row.mode": "Modus", "row.view": "Ansicht", "row.theme": "Thema", "row.speed": "Tempo",
        "row.density": "Dichte", "row.cell": "Zellgröße", "row.pattern": "Muster", "row.info": "Info",
        "row.language": "Sprache",
        "mode.field": "Feld (GPU)", "mode.infinite": "Unendlich",
        "view.single": "Einzeln", "view.tiles": "Kacheln",
        "density.random": "Zufall", "pattern.place": "Setzen",
        "on": "An", "off": "Aus",
        "pause": "Pause", "resume": "Weiter", "new": "Neu", "clear": "Leeren",
        "fit": "Alles zeigen", "fullscreen": "Vollbild", "windowed": "Fenster", "quit": "Beenden",
        "gens_per_s": "{n} Gen/s",
        "hud.generation": "Generation {n}", "hud.paused": "PAUSE", "hud.theme": "Thema {name}",
        "hud.field": "Feld {w}×{h}", "hud.density": "Dichte {p}", "hud.pattern_placed": "Muster",
        "hud.zoom": "Zoom {z}", "hud.view": "Ansicht {name}",
        "hud.infinite": "Unendlich (HashLife)", "hud.cells": "{n} Zellen", "hud.pattern": "Muster {name}",
        "hud.keys": "M Menü · Leertaste Pause · R Neu · A Alles zeigen · Maus Zeichnen · Mausrad Zoom · H Ausblenden",
        "theme.neon": "Neon", "theme.ember": "Glut", "theme.matrix": "Matrix", "theme.pastel": "Pastell",
        "theme.classic": "Klassik",
        "pattern.acorn": "Acorn", "pattern.r-pentomino": "R-Pentomino", "pattern.glider-gun": "Glider-Kanone",
        "pattern.diehard": "Diehard", "pattern.glider": "Glider",
    },
    "en": {
        "menu.title": "Settings",
        "menu.open": "Menu (M)",
        "row.mode": "Mode", "row.view": "View", "row.theme": "Theme", "row.speed": "Speed",
        "row.density": "Density", "row.cell": "Cell size", "row.pattern": "Pattern", "row.info": "Info",
        "row.language": "Language",
        "mode.field": "Field (GPU)", "mode.infinite": "Infinite",
        "view.single": "Single", "view.tiles": "Tiles",
        "density.random": "Random", "pattern.place": "Place",
        "on": "On", "off": "Off",
        "pause": "Pause", "resume": "Resume", "new": "New", "clear": "Clear",
        "fit": "Show all", "fullscreen": "Fullscreen", "windowed": "Window", "quit": "Quit",
        "gens_per_s": "{n} gen/s",
        "hud.generation": "Generation {n}", "hud.paused": "PAUSED", "hud.theme": "Theme {name}",
        "hud.field": "Field {w}×{h}", "hud.density": "Density {p}", "hud.pattern_placed": "Pattern",
        "hud.zoom": "Zoom {z}", "hud.view": "View {name}",
        "hud.infinite": "Infinite (HashLife)", "hud.cells": "{n} cells", "hud.pattern": "Pattern {name}",
        "hud.keys": "M Menu · Space Pause · R New · A Show all · Mouse Draw · Wheel Zoom · H Hide",
        "theme.neon": "Neon", "theme.ember": "Ember", "theme.matrix": "Matrix", "theme.pastel": "Pastel",
        "theme.classic": "Classic",
        "pattern.acorn": "Acorn", "pattern.r-pentomino": "R-pentomino", "pattern.glider-gun": "Glider gun",
        "pattern.diehard": "Diehard", "pattern.glider": "Glider",
    },
}

_lang = "en"


def detect_language() -> str:
    """German if the system locale is German, otherwise English."""
    try:
        name = locale.getlocale()[0] or ""
    except ValueError:
        name = ""
    return "de" if name.lower().startswith(("de", "german")) else "en"


def set_language(lang: str):
    global _lang
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}")
    _lang = lang


def language() -> str:
    return _lang


def tr(key: str, **values) -> str:
    return TEXT[_lang][key].format(**values)


def fmt_int(n: int) -> str:
    """Thousands separators of the current language; huge numbers in scientific notation."""
    if n >= 10 ** 12:
        mantissa, exponent = f"{n:.2e}".split("e+")
        if _lang == "de":
            mantissa = mantissa.replace(".", ",")
        return f"{mantissa}·10^{int(exponent)}"
    text = f"{n:,}"
    return text.replace(",", ".") if _lang == "de" else text
