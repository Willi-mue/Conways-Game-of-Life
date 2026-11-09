import numpy as np
import pytest

import life


def place(pattern, shape=(10, 10), at=(3, 3)):
    grid = np.zeros(shape, np.uint8)
    p = np.array(pattern, np.uint8)
    grid[at[0]:at[0] + p.shape[0], at[1]:at[1] + p.shape[1]] = p
    return grid


BLINKER = [[1, 1, 1]]
BLOCK = [[1, 1], [1, 1]]
GLIDER = [[0, 1, 0],
          [0, 0, 1],
          [1, 1, 1]]


def test_single_cell_dies():
    assert life.step(place([[1]])).sum() == 0


def test_block_is_stable():
    grid = place(BLOCK)
    assert np.array_equal(life.step(grid), grid)


def test_blinker_has_period_two():
    grid = place(BLINKER)
    once = life.step(grid)
    assert not np.array_equal(once, grid)
    assert np.array_equal(life.step(once), grid)


def test_glider_moves_diagonally_after_four_steps():
    grid = place(GLIDER)
    for _ in range(4):
        grid = life.step(grid)
    assert np.array_equal(grid, place(GLIDER, at=(4, 4)))


def test_grid_wraps_around_edges():
    # Glider crossing the corner of a torus reappears on the other side unchanged in shape.
    grid = place(GLIDER, shape=(8, 8), at=(0, 0))
    for _ in range(4 * 8):
        grid = life.step(grid)
    assert np.array_equal(grid, place(GLIDER, shape=(8, 8), at=(0, 0)))


@pytest.mark.parametrize("density", [0.0, 0.25, 1.0])
def test_random_grid_density(density):
    grid = life.random_grid(200, 300, density, np.random.default_rng(1))
    assert grid.shape == (200, 300)
    assert grid.dtype == np.uint8
    assert abs(grid.mean() - density) < 0.01
