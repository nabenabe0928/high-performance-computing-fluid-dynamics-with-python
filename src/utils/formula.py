import numpy as np
from typing import Tuple


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
    def velocity_direction_set():
        """ Note: Those do not have identical norms. """
        return np.array([[0, 0], [1, 0], [0, 1],
                         [-1, 0], [0, -1], [1, 1],
                         [-1, 1], [-1, -1], [1, -1]])


class FluidField2D():
    def __init__(self, X: int, Y: int):
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
        """
        self._pdf = np.zeros((X, Y, 9))
        self._density = np.zeros((X, Y))
        self._velocity = np.zeros((X, Y, 2))
        self._lattice_grid_shape = (X, Y)
        self._finish_initialize = False

    @property
    def pdf(self) -> np.ndarray:
        return self._pdf

    @property
    def density(self) -> np.ndarray:
        return self._density

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def lattice_grid_shape(self) -> Tuple[int, int]:
        return self._lattice_grid_shape

    def init_pdf(self, init_vals: np.ndarray) -> None:
        """ initialize the density """
        assert init_vals.shape == self._pdf.shape
        assert not self._finish_initialize
        self._pdf = init_vals
        self._finish_initialize = True

    def update_density(self) -> None:
        assert self._finish_initialize
        self._density = np.sum(self._pdf, axis=-1)

    def update_velocity_field(self) -> None:
        assert self._finish_initialize

        self._velocity[:, :, 0] = (
            np.sum(self._pdf[:, :, AdjacentIndices.x_right()], axis=-1)
            - np.sum(self._pdf[:, :, AdjacentIndices.x_left()], axis=-1)
        ) / np.maximum(self._density, EPS)

        self._velocity[:, :, 1] = (
            np.sum(self._pdf[:, :, AdjacentIndices.y_upper()], axis=-1)
            - np.sum(self._pdf[:, :, AdjacentIndices.y_lower()], axis=-1)
        ) / np.maximum(self._density, EPS)

    def update_pdf(self) -> None:
        """Update the current pdf based on the streaming operator """
        vs = AdjacentIndices.velocity_direction_set()

        next_pdf = np.zeros_like(self._pdf)
        for i in range(9):
            next_pdf[..., i] = np.roll(self.pdf[..., i], vs[i], axis=(0, 1))

        self._pdf = next_pdf
