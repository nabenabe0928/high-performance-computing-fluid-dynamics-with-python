import numpy as np
from typing import Optional, Tuple
from copy import deepcopy


EPS = 1e-12


class AdjacentIndices():
    """
    The indices for the adjacent cells are:
    y_upper  -> 6 2 5
    y_center -> 3 0 1
    y_lower  -> 7 4 8
    """
    @staticmethod
    def x_left() -> np.ndarray:
        return np.array([3, 6, 7])

    @staticmethod
    def x_center() -> np.ndarray:
        return np.array([0, 2, 4])

    @staticmethod
    def x_right() -> np.ndarray:
        return np.array([1, 5, 8])

    @staticmethod
    def y_upper() -> np.ndarray:
        return np.array([2, 5, 6])

    @staticmethod
    def y_center() -> np.ndarray:
        return np.array([0, 1, 3])

    @staticmethod
    def y_lower() -> np.ndarray:
        return np.array([4, 7, 8])

    @staticmethod
    def velocity_direction_set() -> np.ndarray:
        """ Note: Those do not have identical norms. """
        return np.array([[0, 0], [1, 0], [0, 1],
                         [-1, 0], [0, -1], [1, 1],
                         [-1, 1], [-1, -1], [1, -1]])

    @staticmethod
    def weights() -> np.ndarray:
        """ The weights for each adjacent cell """
        return np.array(
            [4. / 9.]
            + [1. / 9.] * 4
            + [1. / 36.] * 4
        )


class FluidField2D():
    def __init__(self, X: int, Y: int, omega: float = 0.5):
        """
        This class computes and stores
        the density and velocity field
        given (x, y) and initializes
        them at given v and t = 0.
        We update each value every time
        we update().

        Attributes:
            _pdf (np.ndarray):
                probability density function
                of the 8 adjacent cells
                and the target cell.
                shape is (X, Y, 9)

            _density (np.ndarray):
                density of the given location (x, y).
                The shape is (X, Y)

            _velocity (np.ndarray):
                velocity of the given location (x, y).
                The shape is (X, Y, 2).

            _field_shape (Tuple[int, int]) = (X, Y):
                The shape of lattice grid
                X (int): The size of x axis
                Y (int): The size of y axis

            _omega (float):
                relaxation term
        """
        self._pdf = np.zeros((X, Y, 9))
        self._pdf_eq = np.zeros((X, Y, ))
        self._density = np.zeros((X, Y))
        self._velocity = np.zeros((X, Y, 2))
        self._lattice_grid_shape = (X, Y)
        self._finish_initialize = False

        assert 0 < omega < 2
        self._omega = omega

    @property
    def pdf(self) -> np.ndarray:
        return self._pdf

    @pdf.setter
    def pdf(self, vals: np.ndarray) -> None:
        raise NotImplementedError("pdf is not supposed to change from outside.")

    @property
    def pdf_eq(self) -> np.ndarray:
        return self._pdf_eq

    @pdf_eq.setter
    def pdf_eq(self, vals: np.ndarray) -> None:
        raise NotImplementedError("pdf_eq is not supposed to change from outside.")

    @property
    def density(self) -> np.ndarray:
        return self._density

    @density.setter
    def density(self, vals: np.ndarray) -> None:
        raise NotImplementedError("density is not supposed to change from outside.")

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, vals: np.ndarray) -> None:
        raise NotImplementedError("velocity is not supposed to change from outside.")

    @property
    def lattice_grid_shape(self) -> Tuple[int, int]:
        return self._lattice_grid_shape

    @lattice_grid_shape.setter
    def lattice_grid_shape(self, val: Tuple[int, int]) -> None:
        raise NotImplementedError("lattice_grid_shape is not supposed to change from outside.")

    @property
    def omega(self) -> float:
        return self._omega

    @omega.setter
    def omega(self, val: float) -> None:
        raise NotImplementedError("omega is not supposed to change from outside.")

    def _init_pdf(self, init_vals: np.ndarray) -> None:
        assert init_vals.shape == self._pdf.shape
        assert not self._finish_initialize
        self._pdf = init_vals

    def _init_density(self, init_vals: np.ndarray) -> None:
        assert init_vals.shape == self._density.shape
        assert not self._finish_initialize
        self._density = init_vals

    def _init_velocity(self, init_vals: np.ndarray) -> None:
        assert init_vals.shape == self._velocity.shape
        assert not self._finish_initialize
        self._velocity = init_vals

    def init_vals(self, init_pdf: Optional[np.ndarray] = None,
                  init_density: Optional[np.ndarray] = None,
                  init_vel: Optional[np.ndarray] = None) -> None:
        assert not self._finish_initialize

        if init_pdf is not None:
            self._init_pdf(init_pdf)
        if init_density is not None:
            self._init_density(init_density)
        if init_vel is not None:
            self._init_velocity(init_vel)

        self._finish_initialize = True

    def update_density(self) -> None:
        assert self._finish_initialize
        self._density = np.sum(self.pdf, axis=-1)

    def update_velocity(self) -> None:
        assert self._finish_initialize

        self._velocity[:, :, 0] = (
            np.sum(self.pdf[:, :, AdjacentIndices.x_right()], axis=-1)
            - np.sum(self.pdf[:, :, AdjacentIndices.x_left()], axis=-1)
        ) / np.maximum(self.density, EPS)

        self._velocity[:, :, 1] = (
            np.sum(self.pdf[:, :, AdjacentIndices.y_upper()], axis=-1)
            - np.sum(self.pdf[:, :, AdjacentIndices.y_lower()], axis=-1)
        ) / np.maximum(self.density, EPS)

    def update_pdf(self) -> None:
        """Update the current pdf based on the streaming operator """
        vs = AdjacentIndices.velocity_direction_set()

        next_pdf = np.zeros_like(self.pdf)
        for i in range(9):
            next_pdf[..., i] = np.roll(self.pdf[..., i], vs[i], axis=(0, 1))

        self._pdf = next_pdf

    def _apply_local_equilibrium(self) -> None:
        vs = AdjacentIndices.velocity_direction_set()
        # (X, Y, 2) @ (2, 9) -> (X, Y, 9)
        dotprod = self.velocity @ vs.T
        W = AdjacentIndices.weights()
        v_norm2 = np.linalg.norm(self.velocity, axis=-1) ** 2

        self._pdf_eq = W[np.newaxis, np.newaxis, ...] * self.density[..., np.newaxis] * (
            1. + 3. * dotprod + 4.5 * dotprod ** 2 - 1.5 * v_norm2[..., np.newaxis]
        )

    def lattice_boltzmann_step(self, boundary_handling=None) -> None:
        self._apply_local_equilibrium()

        pdf_pre = deepcopy(self.pdf)
        pdf_mid = (self.pdf + (self.pdf_eq - self.pdf) * self._omega)
        self._pdf = deepcopy(pdf_mid)
        self.update_pdf()

        self.update_density()
        self.update_velocity()
