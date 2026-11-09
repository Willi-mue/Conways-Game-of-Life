"""Reference implementation of Conway's Game of Life on the CPU (numpy).

Used to seed the GPU simulation and as the ground truth in tests.
The grid is a torus: cells on one edge neighbour the cells on the opposite edge.
"""
import numpy as np

# Range from which a random start density is drawn when none is given.
DENSITY_RANGE = (0.05, 0.4)


def random_density(rng: np.random.Generator) -> float:
    return float(rng.uniform(*DENSITY_RANGE))


def random_grid(height: int, width: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Grid of shape (height, width) with roughly ``density`` live cells (1) and dead cells (0)."""
    return (rng.random((height, width)) < density).astype(np.uint8)


def step(cells: np.ndarray) -> np.ndarray:
    """Compute the next generation (B3/S23) on a toroidal grid."""
    neighbours = sum(
        np.roll(cells, (dy, dx), axis=(0, 1))
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    return ((neighbours == 3) | ((cells == 1) & (neighbours == 2))).astype(np.uint8)
