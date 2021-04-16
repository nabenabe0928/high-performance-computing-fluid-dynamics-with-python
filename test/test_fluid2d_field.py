import numpy as np
import unittest

from src.utils.formula import FluidField2D


class TestFluidField2D(unittest.TestCase):
    def setUp(self):
        init_vals = np.arange(9)[np.newaxis, np.newaxis, :]
        self.init_vals = np.array(init_vals, dtype=np.float32)

    def test_init_vals(self):
        field = FluidField2D(1, 1)
        init_density = np.random.random((1, 1))
        init_vel = np.random.random((1, 1, 2))
        field.init_vals(init_pdf=self.init_vals,
                        init_density=init_density,
                        init_vel=init_vel)

        self.assertEqual(field.pdf.shape, self.init_vals.shape)
        self.assertEqual(field.velocity.shape, init_vel.shape)
        self.assertEqual(field.density.shape, init_density.shape)
        self.assertAlmostEqual(field.pdf.sum(), self.init_vals.sum(), places=1)
        self.assertAlmostEqual(field.velocity.sum(), init_vel.sum(), places=1)
        self.assertAlmostEqual(field.density.sum(), init_density.sum(), places=1)

    def test_update_density(self):
        field = FluidField2D(1, 1)
        field.init_vals(init_pdf=self.init_vals)
        field.update_density()
        self.assertAlmostEqual(field.density, 36.0)

    def test_update_velocity(self):
        field = FluidField2D(1, 1)
        field.init_vals(init_pdf=self.init_vals)
        field.update_density()
        field.update_velocity()
        ans = np.array([[[- 2.0 / 36.0, - 6.0 / 36.0]]])

        self.assertEqual(field.velocity.shape, ans.shape)
        self.assertAlmostEqual(field.velocity.sum(), ans.sum(), places=1)

    def test_update_pdf(self):
        field = FluidField2D(1, 1)
        field.init_vals(init_pdf=self.init_vals)
        field.update_density()
        field.update_velocity()
        field.update_pdf()
        ans = np.array([[[0., 1., 2., 3., 4., 5., 6., 7., 8.]]])
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(field.pdf.sum(), ans.sum(), places=1)

    def test_apply_local_equilibrium(self):
        field = FluidField2D(1, 1)
        field.init_vals(init_pdf=self.init_vals)
        field._apply_local_equilibrium()
        raise ValueError


if __name__ == '__main__':
    unittest.main()
