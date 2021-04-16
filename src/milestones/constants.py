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


def sinusoidal_velocity(lattice_grid_shape: Tuple[int, int], epsilon: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return initial values according to
    rho(x, y, 0) = 1 and v(x, y, 0) = eps * sin(2PI * y/Y)

    Args:
        epsilon: amplitude of swinging

    Returns:
        density := rho(x, y, 0) = 1
        v(x, y, 0) = eps * sin(2PI * y/Y)

    Note:
        constraint |v| < 0.1
    """
    assert abs(epsilon) < 0.1

    X, Y = lattice_grid_shape
    density = np.ones(lattice_grid_shape)
    vel = np.zeros((*lattice_grid_shape, 2))
    vx = epsilon * np.sin(2 * np.pi * np.arange(Y) / Y)
    vel[..., 0] = np.tile(vx, (X, 1))

    return density, vel


def sinusoidal_density(lattice_grid_shape: Tuple[int, int], epsilon: float,
                       rho0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return initial values according to
    rho(x, y, 0) = rho0 + eps*sin(2PI x/X)
    and v(x, y, 0) = 0

    Args:
        epsilon: amplitude of swinging
        rho0: The offset of density

    Returns:
        density := rho(x, y, 0) = rho0 + eps * sin(2PI x/X)
        v(x, y, 0) = 0
    """
    assert rho0 + epsilon < 1
    assert rho0 - epsilon > 0
    X, Y = lattice_grid_shape
    vel = np.zeros((*lattice_grid_shape, 2))
    density_x = rho0 + epsilon * np.sin(2 * np.pi * np.arange(X) / X)
    density = np.tile(density_x, (Y, 1)).T

    return density, vel
