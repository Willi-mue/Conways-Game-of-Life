"""Minimal clickable in-game menu drawn with pyglet shapes and labels.

The menu is described as rows of items; each item is a button, a text or a spacer.
Texts and "active" states are callables, so ``refresh()`` always shows the current game state.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import pyglet
from pyglet import shapes
from pyglet.graphics import Batch, Group

FONT = "Segoe UI"
COL_PANEL = (14, 16, 24, 230)
COL_BUTTON = (42, 48, 64, 255)
COL_HOVER = (64, 74, 100, 255)
COL_ACTIVE = (0, 140, 190, 255)
COL_TEXT = (235, 238, 245, 255)
COL_LABEL = (150, 158, 175, 255)

Text = Union[str, Callable[[], str]]


@dataclass
class Item:
    kind: str                                   # "button", "text" or "space"
    text: Text = ""
    action: Optional[Callable[[], None]] = None
    active: Optional[Callable[[], bool]] = None
    weight: float = 1.0


def button(text: Text, action, active=None, weight=1.0) -> Item:
    return Item("button", text, action, active, weight)


def text(value: Text, weight=1.0) -> Item:
    return Item("text", value, weight=weight)


def space(weight=1.0) -> Item:
    return Item("space", weight=weight)


def _resolve(value: Text) -> str:
    return value() if callable(value) else value


class _Widget:
    def __init__(self, item: Item, batch: Batch, mid: Group, fg: Group):
        self.item = item
        self.box = (0, 0, 0, 0)
        self.rect = (shapes.RoundedRectangle(0, 0, 1, 1, radius=6, color=COL_BUTTON, batch=batch, group=mid)
                     if item.kind == "button" else None)
        self.label = (pyglet.text.Label("", font_name=FONT, font_size=10.5, color=COL_TEXT, anchor_x="center",
                                        anchor_y="center", batch=batch, group=fg)
                      if item.kind != "space" else None)

    def place(self, x, y, w, h):
        self.box = (x, y, w, h)
        if self.rect:
            self.rect.position = (x, y)
            self.rect.width, self.rect.height = w, h
        if self.label:
            self.label.position = (x + w / 2, y + h / 2, 0)

    def hit(self, px, py):
        x, y, w, h = self.box
        return self.item.kind == "button" and x <= px <= x + w and y <= py <= y + h


class Menu:
    WIDTH = 330
    ROW = 36
    PAD = 14
    LABEL_W = 92
    GAP = 6
    MARGIN = 12

    def __init__(self, title: str, rows, opener_text: str = "Menu (M)"):
        """rows: list of (row label or None, [items])."""
        self.batch = Batch()
        bg, mid, fg = Group(0), Group(1), Group(2)
        self.visible = False
        self.panel = shapes.RoundedRectangle(0, 0, 1, 1, radius=12, color=COL_PANEL, batch=self.batch, group=bg)
        self.title = pyglet.text.Label(title, font_name=FONT, font_size=13, weight="bold", color=COL_TEXT,
                                       anchor_y="center", batch=self.batch, group=fg)
        self.close = _Widget(button("×", self.hide), self.batch, mid, fg)
        self.rows = []
        for label, items in rows:
            row_label = (pyglet.text.Label(label, font_name=FONT, font_size=10.5, color=COL_LABEL,
                                           anchor_y="center", batch=self.batch, group=fg) if label else None)
            self.rows.append((row_label, [_Widget(it, self.batch, mid, fg) for it in items]))
        self.widgets = [self.close] + [w for _, ws in self.rows for w in ws]
        self.hover = None
        self.box = (0, 0, 0, 0)

        # small button shown in the corner while the menu is closed
        self.opener_batch = Batch()
        self.opener = _Widget(button(opener_text, self.show), self.opener_batch, Group(1), Group(2))
        self.opener.rect.color = COL_PANEL

    # --- visibility ------------------------------------------------------------------------

    def show(self):
        self.visible = True
        self.refresh()

    def hide(self):
        self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()

    # --- layout & state ----------------------------------------------------------------------

    def layout(self, win_w, win_h):
        x0 = win_w - self.WIDTH - self.MARGIN
        height = 2 * self.PAD + self.ROW * (len(self.rows) + 1)
        top = win_h - self.MARGIN
        self.box = (x0, top - height, self.WIDTH, height)
        self.panel.position = (x0, top - height)
        self.panel.width, self.panel.height = self.WIDTH, height

        cy = top - self.PAD - self.ROW / 2
        self.title.position = (x0 + self.PAD, cy, 0)
        self.close.place(x0 + self.WIDTH - self.PAD - 28, cy - 13, 28, 26)

        for i, (label, widgets) in enumerate(self.rows, start=1):
            cy = top - self.PAD - self.ROW * i - self.ROW / 2
            left = x0 + self.PAD
            if label:
                label.position = (left, cy, 0)
                left += self.LABEL_W
            right = x0 + self.WIDTH - self.PAD
            total = sum(w.item.weight for w in widgets) or 1
            avail = right - left - self.GAP * (len(widgets) - 1)
            x = left
            for w in widgets:
                width = avail * w.item.weight / total
                w.place(x, cy - 13, width, 26)
                x += width + self.GAP

        self.opener.place(win_w - self.MARGIN - 92, win_h - self.MARGIN - 30, 92, 30)

    def refresh(self):
        for w in self.widgets + [self.opener]:
            if w.label is not None:
                w.label.text = _resolve(w.item.text)
            if w.rect is not None and w is not self.opener:
                if w.item.active and w.item.active():
                    w.rect.color = COL_ACTIVE
                elif w is self.hover:
                    w.rect.color = COL_HOVER
                else:
                    w.rect.color = COL_BUTTON
        self.opener.rect.color = COL_HOVER if self.hover is self.opener else COL_PANEL

    # --- input -------------------------------------------------------------------------------

    def contains(self, px, py):
        """True if the point is on the open menu or on the corner button."""
        return self._inside_panel(px, py) or (not self.visible and self.opener.hit(px, py))

    def _inside_panel(self, px, py):
        x, y, w, h = self.box
        return self.visible and x <= px <= x + w and y <= py <= y + h

    def _clickable(self):
        return self.widgets if self.visible else [self.opener]

    def on_press(self, px, py) -> bool:
        """Handle a click. Returns True if the click belonged to the menu (game should ignore it)."""
        for w in self._clickable():
            if w.hit(px, py):
                w.item.action()
                self.refresh()
                return True
        return self._inside_panel(px, py)

    def on_motion(self, px, py):
        hover = next((w for w in self._clickable() if w.hit(px, py)), None)
        if hover is not self.hover:
            self.hover = hover
            self.refresh()

    def draw(self):
        if self.visible:
            self.batch.draw()
        else:
            self.opener_batch.draw()
