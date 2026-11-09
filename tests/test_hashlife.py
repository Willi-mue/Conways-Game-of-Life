import numpy as np
import pytest

import hashlife as hl
import life


def numpy_run(cells, generations, margin):
    """Reference: numpy on a torus big enough that the pattern never touches the edges."""
    h, w = cells.shape
    grid = np.zeros((h + 2 * margin, w + 2 * margin), np.uint8)
    grid[margin:margin + h, margin:margin + w] = cells
    for _ in range(generations):
        grid = life.step(grid)
    return grid


@pytest.mark.parametrize("generations", [0, 1, 2, 3, 7, 8, 13, 64, 100])
def test_matches_numpy_on_random_soup(generations):
    cells = life.random_grid(20, 24, 0.4, np.random.default_rng(generations))
    margin = generations + 4
    expected = numpy_run(cells, generations, margin)
    root = hl.advance(hl.from_cells(cells, -margin + margin, 0), generations)   # pattern at (0, 0)
    got = hl.to_cells(root, -margin, -margin, expected.shape[1], expected.shape[0])
    assert np.array_equal(got, expected)


def test_advance_in_pieces_equals_advance_at_once():
    root = hl.from_cells(hl.pattern("R-Pentomino"))
    a = hl.advance(hl.advance(hl.advance(root, 100), 23), 377)
    b = hl.advance(root, 500)
    assert a.n == b.n
    assert hl.bounding_box(a) == hl.bounding_box(b)


@pytest.mark.parametrize("name, generations, population", [
    ("Diehard", 129, None),            # still alive one generation before the end
    ("Diehard", 130, 0),               # dies after exactly 130 generations
    ("R-Pentomino", 1103, 116),        # stabilises at 116 cells
    ("Acorn", 5206, 633),              # stabilises at 633 cells
])
def test_known_pattern_facts(name, generations, population):
    root = hl.advance(hl.from_cells(hl.pattern(name)), generations)
    if population is None:
        assert root.n > 0
    else:
        assert root.n == population


def test_glider_gun_adds_one_glider_every_30_generations():
    root = hl.from_cells(hl.pattern("Glider-Kanone"))
    assert root.n == 36
    pops = [hl.advance(root, 300 + 30 * i).n for i in range(4)]
    assert [b - a for a, b in zip(pops, pops[1:])] == [5, 5, 5]


def test_huge_jump_is_fast():
    # a glider after one trillion generations: 2^40 / 4 cells diagonally away, still 5 cells
    root = hl.advance(hl.from_cells(hl.pattern("Glider")), 1 << 40)
    assert root.n == 5
    x0, y0, x1, y1 = hl.bounding_box(root)
    assert x1 - x0 == 3 and y1 - y0 == 3
    assert abs(x0) > (1 << 37)


def test_set_cell_and_rasterize():
    root = hl.empty(3)
    for x, y in [(0, 0), (5, -3), (1000, 2000)]:          # last one forces the universe to grow
        root = hl.set_cell(root, x, y, True)
    assert root.n == 3
    assert hl.bounding_box(root) == (0, -3, 1001, 2001)
    img = hl.rasterize(root, 0, -4, 4, 2, 2)              # 4×2 pixels of 4×4 cells
    # row 0 = cells y -4..-1 (contains (5,-3) in column 1), row 1 = y 0..3 (contains (0,0) in column 0)
    assert img.sum() == 2 and img[1, 0] == 1 and img[0, 1] == 1
    root = hl.set_cell(root, 5, -3, False)
    assert root.n == 2


def test_parse_rle_orientation():
    grid = hl.parse_rle("x = 3, y = 2\n3o$o!")
    # RLE rows go top to bottom; our row 0 is the bottom
    assert grid.tolist() == [[1, 0, 0], [1, 1, 1]]
