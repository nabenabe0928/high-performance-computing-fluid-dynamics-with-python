import numpy as np
import unittest

from src.utils.formula import FluidField2D


class TestFluidField2D(unittest.TestCase):
    def setUp(self):
        init_vals = np.arange(9)[np.newaxis, np.newaxis, :]
        self.init_vals = np.array(init_vals, dtype=np.float32)

    def test_init_pdf(self):
        field = FluidField2D(1, 1)
        field.init_pdf(self.init_vals)
        self.assertEqual(field.pdf.shape, self.init_vals.shape)
        self.assertAlmostEqual(field.pdf.sum(), self.init_vals.sum(), places=1)

    def test_update_density(self):
        field = FluidField2D(1, 1)
        field.init_pdf(self.init_vals)
        field.update_density()
        self.assertAlmostEqual(field.density, 36.0)

    def test_update_velocity_field(self):
        field = FluidField2D(1, 1)
        field.init_pdf(self.init_vals)
        field.update_density()
        field.update_velocity_field()
        ans = np.array([[[- 2.0 / 36.0, - 6.0 / 36.0]]])

        self.assertEqual(field.velocity.shape, ans.shape)
        self.assertAlmostEqual(field.velocity.sum(), ans.sum(), places=1)

    def test_update_pdf(self):
        field = FluidField2D(1, 1)
        field.init_pdf(self.init_vals)
        field.update_density()
        field.update_velocity_field()
        field.update_pdf()
        ans = np.array([[[0., 1., 2., 3., 4., 5., 6., 7., 8.]]])
        self.assertEqual(field.pdf.shape, ans.shape)
        self.assertAlmostEqual(field.pdf.sum(), ans.sum(), places=1)


if __name__ == '__main__':
    unittest.main()
