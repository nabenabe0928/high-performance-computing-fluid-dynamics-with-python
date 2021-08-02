import numpy as np
import unittest

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import (
    RigidWall,
    MovingWall,
    PeriodicBoundaryConditionsWithPressureVariation,
    SequentialBoundaryHandlings
)
from src.utils.constants import DirectionIndicators

from test.constants import TestOutputs
from test.utils import abssum


class TestBoundaryHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice_grid_shape = (3, 3)
        X, Y = self.lattice_grid_shape
        self.init_density = np.ones(self.lattice_grid_shape)
        self.init_vel = np.ones((*self.lattice_grid_shape, 2)) * 5e-2
        self.rigid_boundary_locations = [
            DirectionIndicators.TOP
        ]
        self.moving_boundary_locations = [
            DirectionIndicators.BOTTOM
        ]
        self.pbc_boundary_locations = [
            DirectionIndicators.LEFT,
            DirectionIndicators.RIGHT
        ]
        self.wall_vel = np.array([0.5, 0])
        self.init_pbc_boundary = np.zeros((X, Y))

    def initial_set(self, omega: float = 0.5) -> LatticeBoltzmannMethod:
        field = LatticeBoltzmannMethod(*self.lattice_grid_shape, omega=omega,
                                       init_vel=self.init_vel, init_density=self.init_density)
        rigid_wall = RigidWall(field, boundary_locations=self.rigid_boundary_locations)
        moving_wall = MovingWall(field, boundary_locations=self.moving_boundary_locations, wall_vel=self.wall_vel)
        pbc = PeriodicBoundaryConditionsWithPressureVariation(field, boundary_locations=self.pbc_boundary_locations,
                                                              in_density_factor=(1. + 3e-3) / 3.,
                                                              out_density_factor=1. / 3.)

        return field, rigid_wall, moving_wall, pbc

    def test_rigid_wall(self) -> None:
        field, rigid_wall, _, _ = self.initial_set()
        field.local_equilibrium_pdf_update()

        for _ in range(100):
            field.lattice_boltzmann_step(boundary_handling=SequentialBoundaryHandlings(rigid_wall))
        ans = TestOutputs.pdf_rigid_wall
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)

    def test_moving_wall(self) -> None:
        field, _, moving_wall, _ = self.initial_set()
        field.local_equilibrium_pdf_update()

        for _ in range(100):
            field.lattice_boltzmann_step(boundary_handling=SequentialBoundaryHandlings(moving_wall))
        ans = TestOutputs.pdf_moving_wall
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)

    def test_periodic_boundary_conditions(self) -> None:
        field, _, _, pbc = self.initial_set()
        field.local_equilibrium_pdf_update()

        for _ in range(100):
            field.lattice_boltzmann_step(boundary_handling=SequentialBoundaryHandlings(pbc))
        ans = TestOutputs.pdf_pbc
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
