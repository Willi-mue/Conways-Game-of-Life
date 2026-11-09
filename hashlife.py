"""HashLife: Game of Life on an unbounded plane with memoised quadtrees (Gosper's algorithm).

Identical sub-patterns are stored once (hash-consing) and their future is cached, so regular
patterns can be advanced by millions or billions of generations in one go.

Coordinates: x grows to the right, y grows upwards (same orientation as the GPU field).
The universe root is centred on (0, 0) and keeps that centre forever.
Based on the classic formulation (see e.g. John Williamson's "hashlife" write-up).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


class Node:
    """Quadtree node of level k (2^k × 2^k cells). Children: a = top-left (y high, x low), b = top-right,
    c = bottom-left, d = bottom-right. Leaves (k = 0) are the two singletons ON and OFF."""
    __slots__ = ("k", "a", "b", "c", "d", "n", "_hash")

    def __init__(self, k, a, b, c, d, n, h):
        self.k, self.a, self.b, self.c, self.d, self.n, self._hash = k, a, b, c, d, n, h

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"Node(k={self.k}, n={self.n})"


ON = Node(0, None, None, None, None, 1, 1)
OFF = Node(0, None, None, None, None, 0, 0)
_MASK = (1 << 63) - 1
_join_cache: dict = {}


def join(a: Node, b: Node, c: Node, d: Node) -> Node:
    """The unique node with these four children (hash-consed)."""
    key = (a, b, c, d)
    node = _join_cache.get(key)
    if node is None:
        h = (a.k + 2 + 5131830419411 * a._hash + 3758991985019 * b._hash
             + 8973110871315 * c._hash + 4318490180473 * d._hash) & _MASK
        node = Node(a.k + 1, a, b, c, d, a.n + b.n + c.n + d.n, h)
        _join_cache[key] = node
    return node


@lru_cache(maxsize=None)
def empty(k: int) -> Node:
    if k == 0:
        return OFF
    e = empty(k - 1)
    return join(e, e, e, e)


def centre(m: Node) -> Node:
    """Node one level bigger with m in its centre."""
    z = empty(m.k - 1)
    return join(join(z, z, z, m.a), join(z, z, m.b, z), join(z, m.c, z, z), join(m.d, z, z, z))


def _rule(centre_cell: Node, *neighbours: Node) -> Node:
    total = sum(n.n for n in neighbours)
    return ON if total == 3 or (total == 2 and centre_cell.n) else OFF


def _life_4x4(m: Node) -> Node:
    """Centre 2×2 of a 4×4 node after one generation."""
    a, b, c, d = m.a, m.b, m.c, m.d
    na = _rule(a.d, a.a, a.b, b.a, a.c, b.c, c.a, c.b, d.a)
    nb = _rule(b.c, a.b, b.a, b.b, a.d, b.d, c.b, d.a, d.b)
    nc = _rule(c.b, a.c, a.d, b.c, c.a, d.a, c.c, c.d, d.c)
    nd = _rule(d.a, a.d, b.c, b.d, c.b, d.b, c.d, d.c, d.d)
    return join(na, nb, nc, nd)


_succ_cache: dict = {}


def successor(m: Node, j: int) -> Node:
    """Centre half of m (level k-1) advanced by 2^j generations, 0 <= j <= k-2."""
    key = (m, j)
    result = _succ_cache.get(key)
    if result is not None:
        return result
    if m.n == 0:
        result = m.a
    elif m.k == 2:
        result = _life_4x4(m)
    else:
        a, b, c, d = m.a, m.b, m.c, m.d
        # nine overlapping sub-squares of level k-1
        subs = [
            join(a.a, a.b, a.c, a.d), join(a.b, b.a, a.d, b.c), join(b.a, b.b, b.c, b.d),
            join(a.c, a.d, c.a, c.b), join(a.d, b.c, c.b, d.a), join(b.c, b.d, d.a, d.b),
            join(c.a, c.b, c.c, c.d), join(c.b, d.a, c.d, d.c), join(d.a, d.b, d.c, d.d),
        ]
        if j < m.k - 2:
            # advance each sub-square by the full 2^j and stitch their centres together
            s = [successor(x, j) for x in subs]
            result = join(join(s[0].d, s[1].c, s[3].b, s[4].a), join(s[1].d, s[2].c, s[4].b, s[5].a),
                          join(s[3].d, s[4].c, s[6].b, s[7].a), join(s[4].d, s[5].c, s[7].b, s[8].a))
        else:
            # two half-steps of 2^(k-3) each
            s = [successor(x, j - 1) for x in subs]
            result = join(successor(join(s[0], s[1], s[3], s[4]), j - 1),
                          successor(join(s[1], s[2], s[4], s[5]), j - 1),
                          successor(join(s[3], s[4], s[6], s[7]), j - 1),
                          successor(join(s[4], s[5], s[7], s[8]), j - 1))
    _succ_cache[key] = result
    return result


def _is_padded(m: Node) -> bool:
    """All live cells lie in the centre half of m."""
    return (m.k >= 3 and m.a.n == m.a.d.d.n and m.b.n == m.b.c.c.n
            and m.c.n == m.c.b.b.n and m.d.n == m.d.a.a.n)


def _pad(m: Node) -> Node:
    while not _is_padded(m):
        m = centre(m)
    return m


def advance(root: Node, generations: int) -> Node:
    """Advance by any number of generations (binary decomposition into 2^j jumps)."""
    j = 0
    while generations:
        if generations & 1:
            root = centre(_pad(root))
            while root.k < j + 3:            # enough room for the pattern to grow by 2^j each side
                root = centre(root)
            root = successor(root, j)
        generations >>= 1
        j += 1
    return _pad(root)


def clear_caches():
    """Free memory (all nodes not referenced elsewhere are dropped)."""
    _join_cache.clear()
    _succ_cache.clear()
    empty.cache_clear()
    _bbox.cache_clear()


def cache_size() -> int:
    return len(_join_cache) + len(_succ_cache)


# --- building and reading -----------------------------------------------------------------------

def from_cells(cells: np.ndarray, x0: int = 0, y0: int = 0) -> Node:
    """Root node containing ``cells`` (row 0 = bottom) with its bottom-left cell at (x0, y0)."""
    points = np.argwhere(cells > 0)
    root = empty(3)
    for y, x in points:
        root = set_cell(root, int(x0 + x), int(y0 + y), True)
    return root


def set_cell(root: Node, x: int, y: int, alive: bool) -> Node:
    """New root with cell (x, y) set; grows the universe if needed (path copying, O(levels))."""
    while not (-(1 << (root.k - 1)) <= x < (1 << (root.k - 1)) and -(1 << (root.k - 1)) <= y < (1 << (root.k - 1))):
        root = centre(root)
    return _set(root, x + (1 << (root.k - 1)), y + (1 << (root.k - 1)), alive)


def _set(m: Node, x: int, y: int, alive: bool) -> Node:
    if m.k == 0:
        return ON if alive else OFF
    half = 1 << (m.k - 1)
    right, top = x >= half, y >= half
    x, y = x - half * right, y - half * top
    a, b, c, d = m.a, m.b, m.c, m.d
    if top and not right:
        a = _set(a, x, y, alive)
    elif top:
        b = _set(b, x, y, alive)
    elif not right:
        c = _set(c, x, y, alive)
    else:
        d = _set(d, x, y, alive)
    return join(a, b, c, d)


def to_cells(root: Node, x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """Cells of the window [x0, x0+width) × [y0, y0+height) as 0/1 array (row 0 = bottom)."""
    return (rasterize(root, x0, y0, width, height, 0) > 0).astype(np.uint8)


def rasterize(root: Node, x0: int, y0: int, width: int, height: int, scale: int) -> np.ndarray:
    """Population per pixel of a (height, width) image whose pixels are 2^scale × 2^scale cells.

    Pixel (row r, col q) covers cells x in [x0 + q·2^scale, ...), y in [y0 + r·2^scale, ...).
    x0 and y0 must be multiples of 2^scale. Only non-empty nodes inside the window are visited.
    """
    out = np.zeros((height, width), np.int64)
    size_px = 1 << scale
    half = 1 << (root.k - 1)
    stack = [(root, -half, -half)]       # (node, x of its left edge, y of its bottom edge)
    x1, y1 = x0 + width * size_px, y0 + height * size_px
    while stack:
        m, mx, my = stack.pop()
        size = 1 << m.k
        if m.n == 0 or mx >= x1 or my >= y1 or mx + size <= x0 or my + size <= y0:
            continue
        if size <= size_px:
            out[(my - y0) >> scale, (mx - x0) >> scale] += m.n
            continue
        h = size >> 1
        stack.append((m.a, mx, my + h))
        stack.append((m.b, mx + h, my + h))
        stack.append((m.c, mx, my))
        stack.append((m.d, mx + h, my))
    return out


@lru_cache(maxsize=1 << 20)
def _bbox(m: Node):
    """Bounding box of live cells relative to the node's bottom-left corner, or None."""
    if m.n == 0:
        return None
    if m.k == 0:
        return (0, 0, 1, 1)
    h = 1 << (m.k - 1)
    boxes = []
    for child, dx, dy in ((m.a, 0, h), (m.b, h, h), (m.c, 0, 0), (m.d, h, 0)):
        bb = _bbox(child)
        if bb:
            boxes.append((bb[0] + dx, bb[1] + dy, bb[2] + dx, bb[3] + dy))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def bounding_box(root: Node):
    """(x_min, y_min, x_max_exclusive, y_max_exclusive) of all live cells, or None if empty."""
    bb = _bbox(root)
    if bb is None:
        return None
    half = 1 << (root.k - 1)
    return (bb[0] - half, bb[1] - half, bb[2] - half, bb[3] - half)


# --- patterns -----------------------------------------------------------------------------------

def parse_rle(text: str) -> np.ndarray:
    """Pattern in RLE format -> 0/1 array with row 0 = bottom (so it appears upright on screen)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.startswith("#")]
    if lines and lines[0].startswith("x"):
        lines = lines[1:]
    rows, row, count = [], [], ""
    for ch in "".join(lines):
        if ch.isdigit():
            count += ch
            continue
        n = int(count) if count else 1
        count = ""
        if ch == "b":
            row += [0] * n
        elif ch == "o":
            row += [1] * n
        elif ch == "$":
            rows.append(row)
            rows += [[] for _ in range(n - 1)]
            row = []
        elif ch == "!":
            break
    rows.append(row)
    width = max((len(r) for r in rows), default=0)
    grid = np.zeros((len(rows), width), np.uint8)
    for i, r in enumerate(rows):
        grid[i, :len(r)] = r
    return np.flipud(grid)


PATTERNS = {
    # name: (RLE, description)
    "Acorn": ("bo$3bo$2o2b3o!", "7 Zellen, wächst 5206 Generationen lang"),
    "R-Pentomino": ("b2o$2o$bo!", "5 Zellen, chaotisch für 1103 Generationen"),
    "Glider-Kanone": ("24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$"
                      "10bo5bo7bo$11bo3bo$12b2o!", "Gosper: schießt alle 30 Generationen einen Glider"),
    "Diehard": ("6bo$2o$bo3b3o!", "stirbt nach genau 130 Generationen"),
    "Glider": ("bo$2bo$3o!", "kleinstes Raumschiff"),
}


def pattern(name: str) -> np.ndarray:
    return parse_rle(PATTERNS[name][0])
