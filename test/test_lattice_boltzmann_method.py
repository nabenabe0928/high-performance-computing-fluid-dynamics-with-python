import unittest

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from test.constants import TestInputs, TestOutputs
from test.utils import abssum


class TestLatticeBoltzmannMethod(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice_grid_shape = (3, 3)
        self.init_density = TestInputs.init_density
        self.init_pdf = TestInputs.init_pdf
        self.init_vel = TestInputs.init_vel

    def initial_set(self, omega: float = 0.5) -> LatticeBoltzmannMethod:
        field = LatticeBoltzmannMethod(*self.lattice_grid_shape, omega=omega,
                                       init_pdf=self.init_pdf, init_density=self.init_density,
                                       init_vel=self.init_vel)
        return field

    def test_init_vals(self) -> None:
        field = self.initial_set()

        self.assertEqual(field.pdf.shape, self.init_pdf.shape)
        self.assertEqual(field.velocity.shape, self.init_vel.shape)
        self.assertEqual(field.density.shape, self.init_density.shape)
        self.assertAlmostEqual(abssum(field.pdf, self.init_pdf), 0.0, places=1)
        self.assertAlmostEqual(abssum(field.velocity, self.init_vel), 0.0, places=1)
        self.assertAlmostEqual(abssum(field.density, self.init_density), 0.0, places=1)

    def test_update_density(self) -> None:
        field = self.initial_set()
        field.update_density()
        ans = TestOutputs.density_update
        self.assertAlmostEqual(abssum(field.density, ans), 0.0, places=1)

    def test_update_velocity(self) -> None:
        field = self.initial_set()
        field.update_density()
        field.update_velocity()
        ans = TestOutputs.velocity_update

        self.assertEqual(field.velocity.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.velocity, ans), 0.0, places=1)

    def test_update_pdf(self) -> None:
        field = self.initial_set()
        field.update_density()
        field.update_velocity()
        field.update_pdf()
        ans = TestOutputs.pdf_update

        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)

    def test_apply_local_equilibrium(self) -> None:
        field = self.initial_set()
        field._apply_local_equilibrium()
        ans = TestOutputs.pdf_eq_update

        self.assertEqual(field.pdf_eq.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf_eq, ans), 0.0, places=1)

    def test_lattice_boltzmann_step(self) -> None:
        field = self.initial_set()
        field.lattice_boltzmann_step()

        ans = TestOutputs.pdf_boltzmann
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(abssum(field.pdf, ans), 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
