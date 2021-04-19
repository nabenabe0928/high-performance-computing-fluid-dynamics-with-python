import numpy as np
import unittest

from src.simulation_attributes.formula import FluidField2D
from src.simulation_attributes.boundary_handling import RigidWall, MovingWall, PeriodicBoundaryConditions
from test.constants import TestOutputs
from test.utils import abssum


class TestBoundaryHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice_grid_shape = (3, 3)
        X, Y = self.lattice_grid_shape
        self.init_density = np.ones(self.lattice_grid_shape)
        self.init_vel = np.ones((*self.lattice_grid_shape, 2)) * 5e-2
        self.init_rigid_wall = np.zeros(self.lattice_grid_shape)
        self.init_rigid_wall[:, -1] = np.ones(Y)
        self.init_moving_wall = np.zeros(self.lattice_grid_shape)
        self.init_moving_wall[:, 0] = np.ones(Y)
        self.wall_vel = np.array([50, 0])
        self.init_pbc_boundary = np.zeros((X, Y))
        self.init_pbc_boundary[0, :] = np.ones(Y)
        self.init_pbc_boundary[-1, :] = np.ones(Y)

    def initial_set(self, omega: float = 0.5) -> FluidField2D:
        field = FluidField2D(*self.lattice_grid_shape, omega=omega,
                             init_vel=self.init_vel, init_density=self.init_density)
        rigid_wall = RigidWall(field, init_boundary=self.init_rigid_wall)
        moving_wall = MovingWall(field, init_boundary=self.init_moving_wall, wall_vel=self.wall_vel)
        pbc = PeriodicBoundaryConditions(field, init_boundary=self.init_pbc_boundary,
                                         in_density_factor=1. / 3., out_density_factor=(1. + 3e-3) / 3.)

        return field, rigid_wall, moving_wall, pbc

    def test_rigid_wall(self) -> None:
        field, rigid_wall, _, _ = self.initial_set()
        field.local_equilibrium_pdf_update()

        for _ in range(100):
            field.lattice_boltzmann_step(boundary_handling=rigid_wall.boundary_handling)
        ans = TestOutputs.pdf_rigid_wall
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)

    def test_moving_wall(self) -> None:
        field, _, moving_wall, _ = self.initial_set()
        field.local_equilibrium_pdf_update()

        for _ in range(100):
            field.lattice_boltzmann_step(boundary_handling=moving_wall.boundary_handling)
        ans = TestOutputs.pdf_moving_wall
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)

    def test_periodic_boundary_conditions(self) -> None:
        field, _, _, pbc = self.initial_set()
        field.local_equilibrium_pdf_update()

        for _ in range(100):
            field.lattice_boltzmann_step(boundary_handling=pbc.boundary_handling)
        ans = TestOutputs.pdf_pbc
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
