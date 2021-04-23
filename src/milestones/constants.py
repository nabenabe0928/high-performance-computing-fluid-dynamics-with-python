import numpy as np
from typing import Optional, Tuple


class Milestone2InitVals():
    def test1(self, lattice_grid_shape: Tuple[int, int]
              ) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = lattice_grid_shape
        density = np.ones(lattice_grid_shape) * 0.5
        density[X // 2, Y // 2] = 1.0
        velocity = np.zeros((*lattice_grid_shape, 2))
        return density, velocity

    def test2(self, lattice_grid_shape: Tuple[int, int], seed: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed=seed)

        X, Y = lattice_grid_shape
        density = rng.random(lattice_grid_shape)
        velocity = rng.random((*lattice_grid_shape, 2)) * 0.2 - 0.1
        return density, velocity
