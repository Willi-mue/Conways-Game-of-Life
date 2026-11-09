# TODO

## Erledigt (2026-11-09)

- [x] `requirements.txt` / `requirements-dev.txt` angelegt, Python-Environment `.venv`
- [x] README: Installation, Start, Optionen, Tastenbelegung, Projektaufbau
- [x] Fehlendes Icon → `assets/icon.png` / `assets/icon.ico`
- [x] PyQt5-Version entfernt; `main.py` ist jetzt die OpenGL-Version (pyglet 2)
- [x] Probleme der alten Versionen (vertauschte W/S-Tasten, `windll` nur unter Windows, festes 2880×1620-Fenster, globales `window` bei F11, falsche Kommentare bei +/-) durch die Neufassung behoben
- [x] Simulationslogik ausgelagert: `life.py` (numpy-Referenz), `gpu_life.py` (GPU)
- [x] Zufalls-Initialisierung vektorisiert (`life.random_grid`)
- [x] Grid als Textur rendern statt ein Quad pro Zelle (`DISPLAY_FRAG` in `main.py`)
- [x] Simulation auf der GPU (Fragment-Shader, Ping-Pong) – ca. 12.000 Gen/s bei 1920×1080, ca. 3.600 Gen/s bei 4K (numpy: 50 bzw. 12)
- [x] Maus-Interaktion: Zellen zeichnen/löschen, Zoom um den Cursor, Verschieben
- [x] Startwerte: zufällig per Standard, setzbar per `--density`, `--speed`, `--window`, `--grid`, `--seed`
- [x] Unit-Tests für die Regeln + GPU-vs-numpy-Vergleich (`pytest`, 12 Tests)
- [x] Bildschirmerkennung: Vollbild auf Bildschirmauflösung, Feld = Auflösung / `--cell`, `--screen` für Mehrmonitor
- [x] Zwei Ansichten (`K` / `--view`): „einzeln“ (Feld einmal, Zoom bis Feldgröße, kein Verschieben über den Rand) und „Kacheln“
- [x] App-Identität unter Windows (eigene Taskleisten-Gruppe), Icon in allen Größen, `create_shortcut.ps1`
- [x] Einstellungsmenü im Spiel (`M`): Ansicht, Thema, Tempo, Dichte, Zellgröße, Info, Pause/Neu/Leeren, Vollbild, Beenden
- [x] Optik: Farbthemen (`T`), Nachleucht-Spuren, abgerundete Zellen beim Zoomen, Vignette, Info-Anzeige (`H`)
- [x] Bildwiederholung nur bei Bedarf: pausiert 0 Bilder/s, ~0–1 % CPU (vorher wurde dauerhaft neu gezeichnet)
- [x] GPU mit 32 Zellen pro Texel + bitweisem Nachbarzähler: 4K ~23.000 Gen/s (vorher 3.600), 16384² ~700 Gen/s; Spuren zeitbasiert in eigener Textur
- [x] Modus „Unendlich“ mit HashLife (`hashlife.py`, `U`), eingebaute Muster (Acorn, R-Pentomino, Glider-Kanone, Diehard, Glider), „Alles zeigen“ (`A`)
- [x] Tests: 35 (u. a. GPU-Bitversion an Wortgrenzen/Torus, HashLife vs. numpy, bekannte Musterfakten)

- [x] Info-Anzeige standardmäßig aus (`H`, Menü oder `--hud`)
- [x] Neue README mit Bildern und GIFs (`pics/`), `pyproject.toml` mit pytest-Konfiguration

## Entscheidungen (aus DEV_TODO übernommen)

- Lizenz MIT, © Willi Müller
- OpenGL ist die einzige Variante (PyQt5 entfernt), pyglet 2.x
- Standardwerte: Dichte zufällig 5–40 %, 60 Gen/s, Vollbild in Bildschirmauflösung; alles im Menü änderbar

## Offen

- [ ] Alte Dateien löschen (Löschen war in der Session nicht freigegeben)
  - Approach: `git rm main_openGL.py pics/GL_Screenshot_4K.png` und `rm assets/icon.png tests/conftest.py DEV_TODO.md`
  - Done when: die Dateien sind weg und `pytest` läuft weiterhin (Importpfad kommt jetzt aus `pyproject.toml`).

- [ ] Eigene RLE-Dateien laden
  - Approach: Der Parser existiert bereits (`hashlife.parse_rle`). Fehlt noch: `--pattern datei.rle` und/oder Drag & Drop einer `.rle`-Datei aufs Fenster (`on_file_drop`), danach wie „Setzen“ im Menü.
  - Touches: `main.py`
  - Done when: Eine RLE-Datei von conwaylife.com lässt sich in beiden Modi laden.

- [ ] HashLife in einem Hintergrund-Thread
  - Approach: Sehr große Sprünge in chaotischen Phasen (z. B. Acorn) blockieren kurz die Oberfläche. `hl.advance` in einem Worker-Thread ausführen und das Ergebnis im nächsten `tick` übernehmen.
  - Touches: `main.py`
  - Done when: Menü und Zoom bleiben auch bei 1 Mio. Gen/s auf Acorn flüssig.
