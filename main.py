"""Conway's Game of Life – GPU-accelerated with pyglet 2 / OpenGL 3.3, plus an unbounded HashLife mode.

Run ``python main.py --help`` for options. Controls are listed in README.md.
"""
import argparse
from ctypes import byref
import os
import sys

import numpy as np
import pyglet
from pyglet.gl import (
    GL_BLEND, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_R8, GL_RED, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, GLuint, glActiveTexture, glBindTexture, glDeleteTextures, glDisable,
    glGenTextures, glPixelStorei, glTexImage2D, glTexParameteri, glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.window import key, mouse

import hashlife as hl
import i18n
import life
import ui
from i18n import fmt_int, tr
from gpu_life import FULLSCREEN_VERT, GpuLife, fullscreen_quad

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ID = "WilliMueller.GameOfLife"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def load_icons():
    """All icon sizes, so Windows picks a sharp one for title bar, taskbar and Alt+Tab."""
    return [pyglet.image.load(os.path.join(HERE, "assets", f"icon_{s}.png")) for s in ICON_SIZES]


def register_windows_app():
    """Give the process its own taskbar identity.

    Without this Windows treats the game as "python.exe": it shows Python's icon and groups the
    window with every other Python program instead of as an app of its own.
    """
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)


# --- shaders ---------------------------------------------------------------------------------------

SHADER_COMMON = """#version 330 core
uniform float zoom;           // logical window pixels per cell
uniform float pixel_scale;    // framebuffer pixels per logical pixel (HiDPI)
uniform vec2 resolution;      // framebuffer size in pixels
uniform vec3 bg_color;
uniform vec3 trail_far;       // colour of an almost faded trail
uniform vec3 trail_near;      // colour of a freshly died cell
uniform vec3 alive_color;
out vec4 frag;

vec3 background() {           // background with a soft vignette
    vec2 uv = gl_FragCoord.xy / resolution - 0.5;
    return bg_color * (1.0 - 0.6 * dot(uv, uv)) + bg_color * 0.15;
}

// zoomed in: cells as rounded tiles, empty cells as faint slots
vec3 tiles(vec3 col, vec3 bg, bool filled, vec2 c, float cell_px) {
    if (cell_px < 5.0) return col;
    vec2 f = fract(c) - 0.5;
    float radius = 0.22, gap = 0.07;
    vec2 q = abs(f) - (0.5 - gap - radius);
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
    float mask = 1.0 - smoothstep(-0.75 / cell_px, 0.75 / cell_px, d);
    return mix(bg, filled ? col : bg + vec3(0.035), mask);
}
"""

FIELD_FRAG = SHADER_COMMON + """
uniform usampler2D state;     // 32 cells per texel
uniform sampler2D heat;       // one heat value per cell
uniform vec2 offset;          // cell coordinate at the bottom-left window corner
uniform int tiled;            // 1 = field repeats like tiles, 0 = field shown once

void main() {
    float cell_px = zoom * pixel_scale;
    vec2 c = offset + gl_FragCoord.xy / cell_px;
    ivec2 size = textureSize(heat, 0);
    vec2 sizef = vec2(size);
    vec3 bg = background();

    // single view: area outside the field is dimmed, with a thin glowing frame around the field
    if (tiled == 0 && (any(lessThan(c, vec2(0.0))) || any(greaterThanEqual(c, sizef)))) {
        float dist = length(max(max(-c, c - sizef), 0.0)) * cell_px;   // pixels outside the field
        frag = vec4(mix(mix(bg, trail_near, 0.6), bg * 0.35, smoothstep(1.0, 3.0, dist)), 1.0);
        return;
    }

    ivec2 cell = min(ivec2(floor(mod(c, sizef))), size - 1);          // torus: field repeats
    uint word = texelFetch(state, ivec2(cell.x >> 5, cell.y), 0).r;
    float alive = float((word >> uint(cell.x & 31)) & 1u);
    float h = texelFetch(heat, cell, 0).r;

    vec3 col = mix(trail_far, trail_near, h * h);
    col = mix(bg, col, smoothstep(0.0, 0.35, h));
    col = mix(col, alive_color, alive);
    frag = vec4(tiles(col, bg, h > 0.0 || alive > 0.0, c, cell_px), 1.0);
}
"""

INFINITE_FRAG = SHADER_COMMON + """
uniform sampler2D raster;     // density per raster pixel (one raster pixel = raster_cell × raster_cell cells)
uniform vec2 rel_offset;      // view offset relative to the raster origin, in cells (kept small for float precision)
uniform float raster_cell;

void main() {
    float cell_px = zoom * pixel_scale;
    vec2 c = rel_offset + gl_FragCoord.xy / cell_px;
    ivec2 rp = ivec2(floor(c / raster_cell));
    ivec2 rs = textureSize(raster, 0);
    float v = (all(greaterThanEqual(rp, ivec2(0))) && all(lessThan(rp, rs))) ? texelFetch(raster, rp, 0).r : 0.0;
    vec3 bg = background();
    vec3 col = v > 0.0 ? mix(trail_near, alive_color, v) : bg;
    frag = vec4(raster_cell == 1.0 ? tiles(col, bg, v > 0.0, c, cell_px) : col, 1.0);
}
"""

# name: (background, trail far, trail near, alive)
THEMES = {
    "neon":    ((0.020, 0.025, 0.060), (0.10, 0.05, 0.35), (0.00, 0.80, 1.00), (0.90, 1.00, 1.00)),
    "ember":   ((0.040, 0.012, 0.010), (0.30, 0.02, 0.08), (1.00, 0.45, 0.05), (1.00, 0.96, 0.75)),
    "matrix":  ((0.005, 0.030, 0.010), (0.00, 0.12, 0.05), (0.10, 0.85, 0.35), (0.85, 1.00, 0.85)),
    "pastel":  ((0.960, 0.950, 0.930), (0.85, 0.80, 0.95), (0.95, 0.55, 0.65), (0.20, 0.22, 0.35)),
    "classic": ((0.000, 0.000, 0.000), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00), (1.00, 1.00, 1.00)),
}

# German names still accepted on the command line
THEME_ALIASES = {"glut": "ember", "pastell": "pastel", "klassik": "classic"}
VIEW_ALIASES = {"einzeln": "single", "kacheln": "tiles"}

FIELD, INFINITE = "field", "infinite"
MAX_ZOOM = 64.0
MIN_ZOOM = {FIELD: 0.25, INFINITE: 2.0 ** -60}
MIN_GPS = 1
MAX_GPS = {FIELD: 20000, INFINITE: 2 ** 60}
MAX_STEPS_PER_FRAME = 1000
MAX_CELL = 32
DENSITY_STEP = 0.05
FADE_FRAMES = 150                 # frames to keep drawing after a pause so trails can fade out
HASHLIFE_CACHE_LIMIT = 4_000_000  # memoised nodes before the caches are dropped


def fmt_zoom(z):
    return f"{z:.2g}×" if z >= 1 else f"1:{fmt_int(round(1 / z))}"


class GameOfLife(pyglet.window.Window):
    def __init__(self, args, screen, **kwargs):
        super().__init__(caption="Game of Life", resizable=True, vsync=True, screen=screen, **kwargs)
        self.set_icon(*load_icons())
        if args.fullscreen:
            self.set_fullscreen(True, screen=screen)

        self.rng = np.random.default_rng(args.seed)
        self.fixed_density = args.density
        self.density = 0.0

        self.target_screen = screen
        self.cell = args.cell
        # Field size: explicit, or as many cells as the screen has pixels (divided by the cell size).
        self.sim = GpuLife(*(args.grid or self.grid_for_cell(args.cell)))

        self.field_program = ShaderProgram(Shader(FULLSCREEN_VERT, "vertex"), Shader(FIELD_FRAG, "fragment"))
        self.field_quad = fullscreen_quad(self.field_program)
        self.inf_program = ShaderProgram(Shader(FULLSCREEN_VERT, "vertex"), Shader(INFINITE_FRAG, "fragment"))
        self.inf_quad = fullscreen_quad(self.inf_program)

        self.mode = FIELD
        self.home_zoom = args.cell / self.pixel_scale    # one cell = `cell` physical pixels
        self.zoom = self.home_zoom
        self.offset = [0.0, 0.0]
        self.saved_view = {INFINITE: None}
        self.gps = {FIELD: args.speed, INFINITE: args.speed}   # generations per second per mode
        self._pending = 0.0            # fractional generations carried to the next frame
        self.paused = False
        self.generation = {FIELD: 0, INFINITE: 0}
        self._last_cell = None         # for drawing continuous lines with the mouse
        self.theme_names = list(THEMES)
        self.theme = self.theme_names.index(args.theme)
        self.tiled = args.view == "tiles"
        self.pattern_names = list(hl.PATTERNS)
        self.pattern_index = self.pattern_names.index("acorn")

        # HashLife universe + the raster texture that shows it
        self.root = hl.empty(3)
        self.raster_tex = None
        self.raster_size = (0, 0)
        self.raster_key = None
        self.raster_origin = (0, 0)
        self.raster_cell = 1

        # redraw only when something changed (saves power while paused)
        self.dirty = True
        self.fade_frames = 0

        self.hud_batch = pyglet.graphics.Batch()
        self.hud_bg = pyglet.shapes.Rectangle(0, 0, 1, 1, color=(0, 0, 0, 150), batch=self.hud_batch)
        self.hud = pyglet.text.Label("", font_size=11, multiline=True, width=10_000, anchor_y="top",
                                     color=(235, 235, 235, 255), batch=self.hud_batch)
        self.show_hud = args.hud

        self.menu = None
        self.rebuild_menu()
        self._drag_in_ui = False

        self.clamp_view()
        self.reset()

        rate = getattr(screen.get_mode(), "rate", 0) or 60
        pyglet.clock.schedule_interval(self.tick, 1 / max(rate, 30))
        pyglet.clock.schedule_interval(self.update_hud, 0.25)

    @property
    def pixel_scale(self):
        return self.get_framebuffer_size()[0] / max(self.width, 1)

    def grid_for_cell(self, cell):
        return max(self.target_screen.width // cell, 1), max(self.target_screen.height // cell, 1)

    def changed(self):
        """Something visible changed: draw the next frame."""
        self.dirty = True

    # --- menu --------------------------------------------------------------------------------

    def rebuild_menu(self):
        visible = self.menu.visible if self.menu else False
        self.menu = self.build_menu()
        self.menu.layout(self.width, self.height)
        self.menu.visible = visible
        self.menu.refresh()

    def build_menu(self):
        b, t = ui.button, ui.text
        rows = [
            (tr("row.mode"), [b(tr("mode.field"), lambda: self.set_mode(FIELD), lambda: self.mode == FIELD),
                              b(tr("mode.infinite"), lambda: self.set_mode(INFINITE), lambda: self.mode == INFINITE)]),
        ]
        if self.mode == FIELD:
            rows.append((tr("row.view"), [b(tr("view.single"), lambda: self.set_view(False), lambda: not self.tiled),
                                          b(tr("view.tiles"), lambda: self.set_view(True), lambda: self.tiled)]))
        rows += [
            (tr("row.theme"), [b("‹", lambda: self.cycle_theme(-1), weight=0.6),
                               t(lambda: self.theme_label(), weight=2),
                               b("›", lambda: self.cycle_theme(1), weight=0.6)]),
            (tr("row.speed"), [b("–", lambda: self.change_speed(0.5), weight=0.6),
                               t(lambda: tr("gens_per_s", n=fmt_int(self.gps[self.mode])), weight=2),
                               b("+", lambda: self.change_speed(2), weight=0.6)]),
        ]
        if self.mode == FIELD:
            rows += [
                (tr("row.density"), [b("–", lambda: self.change_density(-1), weight=0.6),
                                     t(lambda: f"{self.density:.0%}", weight=0.9),
                                     b("+", lambda: self.change_density(1), weight=0.6),
                                     b(tr("density.random"), self.use_random_density,
                                       lambda: self.fixed_density is None, weight=1.2)]),
                (tr("row.cell"), [b("–", lambda: self.set_cell_size(self.cell // 2 if self.cell > 1 else 1), weight=0.6),
                                  t(lambda: f"{self.cell} px", weight=2),
                                  b("+", lambda: self.set_cell_size(self.cell * 2), weight=0.6)]),
            ]
        rows += [
            (tr("row.pattern"), [b("‹", lambda: self.cycle_pattern(-1), weight=0.6),
                                 t(lambda: self.pattern_label(), weight=2.2),
                                 b("›", lambda: self.cycle_pattern(1), weight=0.6),
                                 b(tr("pattern.place"), self.place_pattern, weight=1.3)]),
            (tr("row.info"), [b(tr("on"), lambda: self.set_hud(True), lambda: self.show_hud),
                              b(tr("off"), lambda: self.set_hud(False), lambda: not self.show_hud)]),
            (tr("row.language"), [b("Deutsch", lambda: self.set_language("de"), lambda: i18n.language() == "de"),
                                  b("English", lambda: self.set_language("en"), lambda: i18n.language() == "en")]),
            (None, [b(lambda: tr("resume") if self.paused else tr("pause"), self.toggle_pause),
                    b(tr("new"), self.reset), b(tr("clear"), lambda: self.reset(empty=True))]),
            (None, [b(tr("fit"), self.fit_all),
                    b(lambda: tr("windowed") if self.fullscreen else tr("fullscreen"), self.toggle_fullscreen),
                    b(tr("quit"), self.close)]),
        ]
        return ui.Menu(tr("menu.title"), rows, opener_text=tr("menu.open"))

    def theme_label(self):
        return tr(f"theme.{self.theme_names[self.theme]}")

    def pattern_label(self):
        return tr(f"pattern.{self.pattern_names[self.pattern_index]}")

    # --- actions (menu and keyboard) ----------------------------------------------------------

    def set_mode(self, mode):
        if mode == self.mode:
            return
        # each mode keeps its own view
        self.saved_view[self.mode] = (self.zoom, list(self.offset))
        self.mode = mode
        self._pending = 0.0
        if mode == INFINITE and self.root.n == 0 and self.generation[INFINITE] == 0:
            self.place_pattern()                    # first visit: start with the selected pattern
        elif self.saved_view.get(mode):
            self.zoom, self.offset = self.saved_view[mode][0], list(self.saved_view[mode][1])
        self.clamp_view()
        self.rebuild_menu()
        self.update_hud()
        self.changed()

    def set_language(self, lang):
        i18n.set_language(lang)
        self.rebuild_menu()
        self.update_hud()
        self.changed()

    def set_view(self, tiled):
        self.tiled = tiled
        self.clamp_view()
        self.changed()

    def set_hud(self, on):
        self.show_hud = on
        self.changed()

    def cycle_theme(self, direction):
        self.theme = (self.theme + direction) % len(self.theme_names)
        self.changed()

    def cycle_pattern(self, direction):
        self.pattern_index = (self.pattern_index + direction) % len(self.pattern_names)

    def change_speed(self, factor):
        self.gps[self.mode] = int(min(max(self.gps[self.mode] * factor, MIN_GPS), MAX_GPS[self.mode]))

    def change_density(self, direction):
        """Fixed density in 5 % steps, starting from the current one; restarts the field."""
        current = self.fixed_density if self.fixed_density is not None else self.density
        steps = round(current / DENSITY_STEP) + direction
        self.fixed_density = round(min(max(steps, 1), round(0.95 / DENSITY_STEP)) * DENSITY_STEP, 2)
        self.reset()

    def use_random_density(self):
        self.fixed_density = None
        self.reset()

    def set_cell_size(self, cell):
        """New field size so that the field still fills the screen with cells of `cell` pixels."""
        cell = min(max(cell, 1), MAX_CELL)
        if cell == self.cell:
            return
        self.cell = cell
        self.sim.delete()
        self.sim = GpuLife(*self.grid_for_cell(cell))
        self.home_zoom = cell / self.pixel_scale
        self.zoom, self.offset = self.home_zoom, [0.0, 0.0]
        self.clamp_view()
        self.reset()

    def toggle_pause(self):
        self.paused = not self.paused
        self.fade_frames = FADE_FRAMES
        self.changed()

    def toggle_fullscreen(self):
        self.set_fullscreen(not self.fullscreen)

    def reset(self, empty=False):
        """Field: new random (or empty) field. Unbounded: back to the selected pattern (or empty)."""
        self.generation[self.mode] = 0
        if self.mode == INFINITE:
            if empty:
                self.root = hl.empty(3)
                hl.clear_caches()
            else:
                self.place_pattern()
            self.changed()
            return
        if empty:
            self.density = 0.0
            cells = np.zeros((self.sim.height, self.sim.width), np.uint8)
        else:
            self.density = self.fixed_density if self.fixed_density is not None else life.random_density(self.rng)
            cells = life.random_grid(self.sim.height, self.sim.width, self.density, self.rng)
        self.sim.upload(cells)
        self.fade_frames = FADE_FRAMES
        self.update_hud()
        self.changed()

    def place_pattern(self):
        cells = hl.pattern(self.pattern_names[self.pattern_index])
        self.generation[self.mode] = 0
        if self.mode == INFINITE:
            hl.clear_caches()
            self.root = hl.from_cells(cells, -(cells.shape[1] // 2), -(cells.shape[0] // 2))
            self.fit_all(min_size=160)
        else:
            field = np.zeros((self.sim.height, self.sim.width), np.uint8)
            h, w = min(cells.shape[0], self.sim.height), min(cells.shape[1], self.sim.width)
            y0, x0 = (self.sim.height - h) // 2, (self.sim.width - w) // 2
            field[y0:y0 + h, x0:x0 + w] = cells[:h, :w]
            self.density = 0.0
            self.sim.upload(field)
            self.fade_frames = FADE_FRAMES
        self.update_hud()
        self.changed()

    def fit_all(self, min_size=0):
        """Zoom so that everything is visible: the whole field, or all live cells of the unbounded universe."""
        if self.mode == FIELD:
            box = (0, 0, self.sim.width, self.sim.height)
        else:
            box = hl.bounding_box(self.root) or (-50, -50, 50, 50)
        x0, y0, x1, y1 = box
        w, h = max(x1 - x0, min_size, 1), max(y1 - y0, min_size, 1)
        self.zoom = min(max(min(self.width / w, self.height / h) * (1.0 if self.mode == FIELD else 0.9),
                            MIN_ZOOM[self.mode]), MAX_ZOOM)
        self.offset = [(x0 + x1) / 2 - self.width / 2 / self.zoom, (y0 + y1) / 2 - self.height / 2 / self.zoom]
        self.clamp_view()
        self.update_hud()
        self.changed()

    # --- simulation --------------------------------------------------------------------------

    def tick(self, dt):
        """Called once per display frame: advance the simulation and draw only if needed."""
        advanced = False
        if not self.paused:
            self._pending += dt * self.gps[self.mode]
            if self.mode == FIELD:
                steps = min(int(self._pending), MAX_STEPS_PER_FRAME)
                self._pending = min(self._pending - steps, 1.0)
            else:
                steps = int(self._pending)
                self._pending -= steps
            if steps:
                self.advance(steps)
                advanced = True
        if advanced or self.dirty or (self.mode == FIELD and self.fade_frames > 0):
            if not advanced and self.fade_frames > 0:
                self.fade_frames -= 1
            self.dirty = False
            self.draw(dt)

    def advance(self, generations):
        if self.mode == FIELD:
            self.sim.step(generations)
            glViewport(0, 0, *self.get_framebuffer_size())
        else:
            self.root = hl.advance(self.root, generations)
            if hl.cache_size() > HASHLIFE_CACHE_LIMIT:
                hl.clear_caches()
        self.generation[self.mode] += generations

    # --- HUD ---------------------------------------------------------------------------------

    def update_hud(self, _dt=None):
        status = tr("hud.paused") if self.paused else tr("gens_per_s", n=fmt_int(self.gps[self.mode]))
        sep = "   ·   "
        if self.mode == FIELD:
            content = tr("hud.pattern_placed") if self.density == 0 else tr("hud.density", p=f"{self.density:.0%}")
            view = tr("view.tiles") if self.tiled else tr("view.single")
            line2 = sep.join([tr("hud.field", w=self.sim.width, h=self.sim.height), content,
                              tr("hud.zoom", z=fmt_zoom(self.zoom / self.home_zoom)), tr("hud.view", name=view)])
        else:
            line2 = sep.join([tr("hud.infinite"), tr("hud.cells", n=fmt_int(self.root.n)),
                              tr("hud.zoom", z=fmt_zoom(self.zoom)), tr("hud.pattern", name=self.pattern_label())])
        line1 = sep.join([tr("hud.generation", n=fmt_int(self.generation[self.mode])), status,
                          tr("hud.theme", name=self.theme_label())])
        text = f"{line1}\n{line2}\n{tr('hud.keys')}"
        if text != self.hud.text:
            self.hud.text = text
            self.layout_hud()
            if self.show_hud:
                self.changed()
        if self.menu is not None and self.menu.visible:
            self.menu.refresh()          # keep values like the speed in sync with keyboard changes

    def layout_hud(self):
        pad, margin = 10, 12
        self.hud.x = margin + pad
        self.hud.y = self.height - margin - pad
        self.hud_bg.x = margin
        self.hud_bg.y = self.height - margin - self.hud.content_height - 2 * pad
        self.hud_bg.width = self.hud.content_width + 2 * pad
        self.hud_bg.height = self.hud.content_height + 2 * pad

    def on_resize(self, width, height):
        result = super().on_resize(width, height)
        if not hasattr(self, "hud"):     # event arrived while the window is still being set up
            return result
        self.clamp_view()
        self.layout_hud()
        self.menu.layout(width, height)
        self.changed()
        return result

    def on_expose(self):
        self.changed()

    # --- rendering ---------------------------------------------------------------------------

    def set_common_uniforms(self, p, fb_w, fb_h):
        bg, far, near, alive = THEMES[self.theme_names[self.theme]]
        values = {"zoom": self.zoom, "pixel_scale": fb_w / max(self.width, 1), "resolution": (fb_w, fb_h),
                  "bg_color": bg, "trail_far": far, "trail_near": near, "alive_color": alive}
        for name, value in values.items():
            if name in p.uniforms:      # the driver removes uniforms a shader doesn't use
                p[name] = value

    def on_draw(self):
        fb_w, fb_h = self.get_framebuffer_size()
        if self.mode == FIELD:
            self.sim.update_heat()               # one fade step of the trails per drawn frame
        glViewport(0, 0, fb_w, fb_h)
        self.clear()
        glDisable(GL_BLEND)
        if self.mode == FIELD:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.sim.state_texture)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.sim.heat_texture)
            p = self.field_program
            p.use()
            self.set_common_uniforms(p, fb_w, fb_h)
            p["state"], p["heat"] = 0, 1
            p["offset"] = tuple(self.offset)
            p["tiled"] = int(self.tiled)
            self.field_quad.draw(GL_TRIANGLES)
        else:
            self.update_raster()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.raster_tex)
            p = self.inf_program
            p.use()
            self.set_common_uniforms(p, fb_w, fb_h)
            p["raster"] = 0
            p["rel_offset"] = (self.offset[0] - self.raster_origin[0], self.offset[1] - self.raster_origin[1])
            p["raster_cell"] = float(self.raster_cell)
            self.inf_quad.draw(GL_TRIANGLES)
        p.stop()
        glActiveTexture(GL_TEXTURE0)
        if self.show_hud:
            self.hud_batch.draw()
        self.menu.draw()

    def update_raster(self):
        """Render the visible part of the HashLife universe into a small texture (one texel per screen
        pixel when zoomed out, one per cell when zoomed in)."""
        scale = max(0, int(np.ceil(np.log2(1 / self.zoom))) if self.zoom < 1 else 0)
        cell = 1 << scale
        x0 = int(np.floor(self.offset[0] / cell)) * cell
        y0 = int(np.floor(self.offset[1] / cell)) * cell
        w = int(np.ceil(self.width / (self.zoom * cell))) + 2
        h = int(np.ceil(self.height / (self.zoom * cell))) + 2
        key_ = (self.root, x0, y0, w, h, scale)
        if key_ == self.raster_key:
            return
        self.raster_key = key_
        pop = hl.rasterize(self.root, x0, y0, w, h, scale)
        # brightness grows with the share of live cells in a pixel; any live cell is clearly visible
        density = pop / float(cell * cell)
        img = np.where(pop > 0, np.clip(0.35 + 0.65 * np.sqrt(density), 0, 1) * 255, 0).astype(np.uint8)
        img = np.ascontiguousarray(img)
        if self.raster_tex is None:
            tex = GLuint()
            glGenTextures(1, byref(tex))
            self.raster_tex = tex
            glBindTexture(GL_TEXTURE_2D, tex)
            for param, value in ((GL_TEXTURE_MIN_FILTER, GL_NEAREST), (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
                                 (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
                glTexParameteri(GL_TEXTURE_2D, param, value)
        glBindTexture(GL_TEXTURE_2D, self.raster_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, img.ctypes.data)
        self.raster_origin = (x0, y0)
        self.raster_cell = cell

    # --- view helpers ------------------------------------------------------------------------

    def cell_at(self, x, y):
        return (int(np.floor(self.offset[0] + x / self.zoom)),
                int(np.floor(self.offset[1] + y / self.zoom)))

    @property
    def fit_zoom(self):
        """Zoom at which the whole field just fits into the window."""
        return min(self.width / self.sim.width, self.height / self.sim.height)

    @property
    def single_field_view(self):
        return self.mode == FIELD and not self.tiled

    @property
    def min_zoom(self):
        return min(self.fit_zoom, MAX_ZOOM) if self.single_field_view else MIN_ZOOM[self.mode]

    def clamp_view(self):
        """Single view: no zooming out past the whole field, and the field never leaves the window.

        If the field is smaller than the window along an axis it is centred on that axis.
        """
        if not self.single_field_view:
            return
        self.zoom = min(max(self.zoom, self.min_zoom), MAX_ZOOM)
        for axis, (window_px, cells) in enumerate(((self.width, self.sim.width), (self.height, self.sim.height))):
            visible = window_px / self.zoom              # number of cells that fit on this axis
            if visible >= cells:
                self.offset[axis] = (cells - visible) / 2
            else:
                self.offset[axis] = min(max(self.offset[axis], 0.0), cells - visible)

    def set_zoom(self, new_zoom, anchor_x=None, anchor_y=None):
        """Zoom while keeping the cell under the anchor point (default: window centre) fixed."""
        new_zoom = min(max(new_zoom, self.min_zoom), MAX_ZOOM)
        ax = self.width / 2 if anchor_x is None else anchor_x
        ay = self.height / 2 if anchor_y is None else anchor_y
        self.offset[0] += ax / self.zoom - ax / new_zoom
        self.offset[1] += ay / self.zoom - ay / new_zoom
        self.zoom = new_zoom
        self.clamp_view()
        self.update_hud()
        self.changed()

    def pan(self, dx_cells, dy_cells):
        self.offset[0] += dx_cells
        self.offset[1] += dy_cells
        self.clamp_view()
        self.changed()

    def paint_line(self, start, end, alive):
        """Set every cell on the straight line from start to end (so fast mouse moves leave no gaps)."""
        (x0, y0), (x1, y1) = start, end
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        points = {(round(x0 + (x1 - x0) * i / n), round(y0 + (y1 - y0) * i / n)) for i in range(n + 1)}
        if self.mode == INFINITE:
            for x, y in points:
                self.root = hl.set_cell(self.root, x, y, alive)
        else:
            if not self.tiled:   # single view: clicks beside the field do nothing
                points = {(x, y) for x, y in points if 0 <= x < self.sim.width and 0 <= y < self.sim.height}
            self.sim.set_cells(points, alive)
            self.fade_frames = FADE_FRAMES
        self.changed()

    # --- input -------------------------------------------------------------------------------

    def on_key_press(self, symbol, modifiers):
        if symbol in (key.M, key.TAB):
            self.menu.toggle()
        elif symbol in (key.SPACE, key.E):
            self.toggle_pause()
        elif symbol == key.N and self.paused:
            self.advance(1)
        elif symbol == key.W:
            self.change_speed(2)
        elif symbol == key.S:
            self.change_speed(0.5)
        elif symbol == key.R:
            self.reset()
        elif symbol == key.C:
            self.reset(empty=True)
        elif symbol == key.T:
            self.cycle_theme(1)
        elif symbol == key.H:
            self.set_hud(not self.show_hud)
        elif symbol == key.K and self.mode == FIELD:
            self.set_view(not self.tiled)
        elif symbol == key.U:
            self.set_mode(INFINITE if self.mode == FIELD else FIELD)
        elif symbol == key.A:
            self.fit_all()
        elif symbol in (key.PLUS, key.NUM_ADD, key.EQUAL):
            self.set_zoom(self.zoom * 2)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            self.set_zoom(self.zoom / 2)
        elif symbol == key.HOME:
            if self.mode == FIELD:
                self.zoom, self.offset = self.home_zoom, [0.0, 0.0]
                self.clamp_view()
            else:
                self.fit_all()
        elif symbol in (key.LEFT, key.RIGHT, key.UP, key.DOWN):
            step = 50 / self.zoom
            self.pan({key.LEFT: -step, key.RIGHT: step}.get(symbol, 0),
                     {key.DOWN: -step, key.UP: step}.get(symbol, 0))
        elif symbol == key.F11:
            self.toggle_fullscreen()
        elif symbol == key.ESCAPE:
            if self.menu.visible:          # Esc closes the menu first, a second Esc quits
                self.menu.hide()
                self.changed()
            else:
                self.close()
            return pyglet.event.EVENT_HANDLED
        self.update_hud()
        self.menu.refresh()
        self.changed()

    def on_mouse_motion(self, x, y, dx, dy):
        hover_before = self.menu.hover
        self.menu.on_motion(x, y)
        if self.menu.hover is not hover_before:
            self.changed()

    def on_mouse_press(self, x, y, button, modifiers):
        self._drag_in_ui = self.menu.on_press(x, y) if button == mouse.LEFT else self.menu.contains(x, y)
        if self._drag_in_ui:
            self.update_hud()
            self.changed()
            return
        if button in (mouse.LEFT, mouse.RIGHT):
            self._last_cell = self.cell_at(x, y)
            self.paint_line(self._last_cell, self._last_cell, alive=button == mouse.LEFT)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._drag_in_ui:               # drag that started on the menu must not paint
            return
        if buttons & mouse.MIDDLE:
            self.pan(-dx / self.zoom, -dy / self.zoom)
        elif buttons & (mouse.LEFT | mouse.RIGHT):
            cell = self.cell_at(x, y)
            self.paint_line(self._last_cell or cell, cell, alive=bool(buttons & mouse.LEFT))
            self._last_cell = cell

    def on_mouse_release(self, x, y, button, modifiers):
        self._last_cell = None
        self._drag_in_ui = False

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y:
            self.set_zoom(self.zoom * (2 ** (0.25 * scroll_y)), x, y)

    def on_close(self):
        if self.raster_tex is not None:
            glDeleteTextures(1, byref(self.raster_tex))
        super().on_close()


def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("format: WIDTHxHEIGHT, e.g. 1920x1080")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be > 0")
    return w, h


def parse_density(text):
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("density must be between 0 and 1")
    return value


def parse_theme(text):
    name = THEME_ALIASES.get(text.lower(), text.lower())
    if name not in THEMES:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(THEMES)}")
    return name


def parse_view(text):
    name = VIEW_ALIASES.get(text.lower(), text.lower())
    if name not in ("single", "tiles"):
        raise argparse.ArgumentTypeError("choose from single, tiles")
    return name


def parse_args(argv=None):
    lo, hi = life.DENSITY_RANGE
    p = argparse.ArgumentParser(description="Conway's Game of Life (GPU/OpenGL + HashLife)")
    p.add_argument("--density", type=parse_density, default=None,
                   help=f"start density 0..1 (default: random between {lo:.0%}% and {hi:.0%}% on every reset)")
    p.add_argument("--speed", type=int, default=60,
                   help=f"generations per second, {MIN_GPS}..{MAX_GPS[FIELD]} (default: 60)")
    p.add_argument("--cell", type=int, default=1,
                   help="cell size in screen pixels; the field still fills the screen (default: 1)")
    p.add_argument("--grid", type=parse_size, default=None,
                   help="field size in cells, overrides --cell (default: screen resolution / cell size)")
    p.add_argument("--window", type=parse_size, default=None,
                   help="start in a window of WIDTHxHEIGHT instead of fullscreen")
    p.add_argument("--screen", type=int, default=None, help="monitor number (default: primary monitor)")
    p.add_argument("--theme", type=parse_theme, default="neon", metavar="{" + ",".join(THEMES) + "}",
                   help="colour theme (default: neon)")
    p.add_argument("--view", type=parse_view, default="single", metavar="{single,tiles}",
                   help="single = field shown once; tiles = field repeats (default: single, toggle with K)")
    p.add_argument("--lang", choices=i18n.LANGUAGES, default=None,
                   help="interface language (default: from the system language)")
    p.add_argument("--hud", action="store_true", help="show the info overlay at start (default: off, toggle with H)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible fields")
    args = p.parse_args(argv)
    args.speed = min(max(args.speed, MIN_GPS), MAX_GPS[FIELD])
    args.cell = max(args.cell, 1)
    args.fullscreen = args.window is None
    i18n.set_language(args.lang or i18n.detect_language())
    return args


def pick_screen(index):
    display = pyglet.display.get_display()
    screens = display.get_screens()
    if index is None:
        return display.get_default_screen()
    if not 0 <= index < len(screens):
        raise SystemExit(f"monitor {index} does not exist (available: 0..{len(screens) - 1})")
    return screens[index]


def main(argv=None):
    args = parse_args(argv)
    register_windows_app()
    screen = pick_screen(args.screen)
    # Windowed size; also the size the window returns to when leaving fullscreen with F11.
    w, h = args.window or (int(screen.width * 0.8), int(screen.height * 0.8))
    GameOfLife(args, screen, width=w, height=h)
    pyglet.app.run(interval=None)    # no fixed redraw: GameOfLife.tick draws only when something changed


if __name__ == "__main__":
    main()
