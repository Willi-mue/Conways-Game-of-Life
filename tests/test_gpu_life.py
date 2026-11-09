"""Checks that the bit-packed GPU shader produces exactly the same generations as the numpy reference."""
import numpy as np
import pytest

import life

pyglet = pytest.importorskip("pyglet")


@pytest.fixture(scope="module")
def gl_context():
    try:
        window = pyglet.window.Window(width=64, height=64, visible=False)
    except Exception as exc:  # no display / no OpenGL 3.3
        pytest.skip(f"no OpenGL context available: {exc}")
    yield window
    window.close()


def test_pack_roundtrip():
    from gpu_life import pack, unpack

    cells = life.random_grid(7, 77, 0.5, np.random.default_rng(0))
    assert np.array_equal(unpack(pack(cells, 3), 77), cells)


# widths: exact multiples of 32, a partial last word, a single partial word, a single full word
@pytest.mark.parametrize("shape", [(64, 64), (37, 53), (20, 7), (16, 32), (33, 100)])
def test_gpu_matches_numpy(gl_context, shape):
    from gpu_life import GpuLife

    h, w = shape
    cells = life.random_grid(h, w, 0.3, np.random.default_rng(42))
    sim = GpuLife(w, h)
    try:
        sim.upload(cells)
        assert np.array_equal(sim.read(), cells)
        for _ in range(25):
            cells = life.step(cells)
            sim.step(1)
            assert np.array_equal(sim.read(), cells)
        sim.step(10)                         # several generations in one call
        for _ in range(10):
            cells = life.step(cells)
        assert np.array_equal(sim.read(), cells)
    finally:
        sim.delete()


def test_glider_crosses_word_and_torus_edges(gl_context):
    from gpu_life import GpuLife

    h, w = 12, 45                            # 2 words, last one partial
    cells = np.zeros((h, w), np.uint8)
    cells[5:8, 28:31] = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
    sim = GpuLife(w, h)
    try:
        sim.upload(cells)
        for _ in range(4 * 45):              # long enough to cross the word border and wrap around
            cells = life.step(cells)
        sim.step(4 * 45)
        assert np.array_equal(sim.read(), cells)
    finally:
        sim.delete()


def test_trail_heat_fades_to_zero(gl_context):
    """Heat of dead cells must reach exactly 0, not get stuck by 8-bit rounding."""
    from gpu_life import GpuLife

    sim = GpuLife(8, 8)
    try:
        cells = np.zeros((8, 8), np.uint8)
        cells[4, 4] = 1                      # lone cell dies in the first generation
        sim.upload(cells)
        sim.update_heat()
        assert sim.read_heat()[4, 4] == 255  # alive -> full heat
        sim.step(1)
        sim.update_heat()
        first = int(sim.read_heat()[4, 4])
        assert 0 < first < 255
        for _ in range(200):
            sim.update_heat()
        assert sim.read_heat()[4, 4] == 0
    finally:
        sim.delete()


def test_set_cells(gl_context):
    from gpu_life import GpuLife

    sim = GpuLife(40, 8)
    try:
        sim.upload(np.zeros((8, 40), np.uint8))
        sim.set_cells([(2, 5), (-1, -1), (33, 0), (31, 0)], True)   # (-1,-1) wraps to (39, 7)
        grid = sim.read()
        assert grid[5, 2] == grid[7, 39] == grid[0, 33] == grid[0, 31] == 1 and grid.sum() == 4
        sim.set_cell(33, 0, False)
        assert sim.read().sum() == 3
    finally:
        sim.delete()
