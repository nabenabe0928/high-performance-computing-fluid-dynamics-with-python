import numpy as np
import unittest

from src.milestones.constants import sinusoidal_density, sinusoidal_velocity
from test.utils import abssum


class SinusoidalVelocity(unittest.TestCase):
    def test_sinusoidal_velocity(self):
        lattice_grid_shape = (3, 3)
        ans = np.array([0.0, 0.008660254037844387, -0.008660254037844384] * 3).reshape(lattice_grid_shape)
        ans = np.dstack([ans, np.zeros(lattice_grid_shape)])
        density, vel = sinusoidal_velocity(lattice_grid_shape=(3, 3), epsilon=0.01)
        self.assertAlmostEqual(abssum(density, np.ones(lattice_grid_shape)), 0.0, places=1)
        self.assertAlmostEqual(abssum(vel, ans), 0.0, places=1)

    def test_sinusoidal_density(self):
        lattice_grid_shape = (3, 3)
        ans = np.array([0.5, 0.50866025, 0.49133975] * 3).reshape(lattice_grid_shape).T
        density, vel = sinusoidal_density(lattice_grid_shape=(3, 3), epsilon=0.01, rho0=0.5)
        self.assertAlmostEqual(abssum(density, ans), 0.0, places=1)
        self.assertAlmostEqual(abssum(vel, np.zeros((*lattice_grid_shape, 2))), 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
