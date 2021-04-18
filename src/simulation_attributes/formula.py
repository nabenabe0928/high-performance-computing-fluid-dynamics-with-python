import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from copy import deepcopy


EPS = 1e-12


def local_equilibrium(velocity: np.ndarray, density: np.ndarray) -> np.ndarray:
    """ The local relaxation of the probability density function """

    assert density.shape == velocity.shape[:-1]

    vs = AdjacentAttributes.velocity_direction_set
    # (X, Y, 2) @ (2, 9) -> (X, Y, 9)
    vel_dot_vs = velocity @ vs.T
    W = AdjacentAttributes.weights
    v_norm2 = np.linalg.norm(velocity, axis=-1) ** 2

    pdf_eq = W[np.newaxis, np.newaxis, ...] * density[..., np.newaxis] * (
        1. + 3. * vel_dot_vs + 4.5 * vel_dot_vs ** 2 - 1.5 * v_norm2[..., np.newaxis]
    )

    return pdf_eq


class MetaAdjacentAttributes(type):
    """
    The attributes for the adjacent cells.
    such as the following indices are:
    y_upper  -> 6 2 5
    y_center -> 3 0 1
    y_lower  -> 7 4 8
    """
    def __init__(cls, *args: List[Any], **kwargs: Dict[str, Any]):
        pass

    @property
    def x_left(cls) -> np.ndarray:
        return np.array([3, 6, 7])

    @property
    def x_center(cls) -> np.ndarray:
        return np.array([0, 2, 4])

    @property
    def x_right(cls) -> np.ndarray:
        return np.array([1, 5, 8])

    @property
    def y_upper(cls) -> np.ndarray:
        return np.array([2, 5, 6])

    @property
    def y_center(cls) -> np.ndarray:
        return np.array([0, 1, 3])

    @property
    def y_lower(cls) -> np.ndarray:
        return np.array([4, 7, 8])

    @property
    def velocity_direction_set(cls) -> np.ndarray:
        """ Note: Those do not have identical norms. """
        return np.array([[0, 0], [1, 0], [0, 1],
                         [-1, 0], [0, -1], [1, 1],
                         [-1, 1], [-1, -1], [1, -1]])

    @property
    def reflected_direction(cls) -> np.ndarray:
        return np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    @property
    def weights(cls) -> np.ndarray:
        """ The weights for each adjacent cell """
        return np.array(
            [4. / 9.]
            + [1. / 9.] * 4
            + [1. / 36.] * 4
        )


class AdjacentAttributes(metaclass=MetaAdjacentAttributes):
    """ From this class, you can call properties above """
    pass


class FluidField2D():
    def __init__(self, X: int, Y: int, omega: float = 0.5,
                 init_pdf: Optional[np.ndarray] = None,
                 init_density: Optional[np.ndarray] = None,
                 init_vel: Optional[np.ndarray] = None):
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
        self._pdf_eq = np.zeros((X, Y, 9))
        self._density = np.zeros((X, Y))
        self._velocity = np.zeros((X, Y, 2))
        self._lattice_grid_shape = (X, Y)
        self._finish_initialize = False

        self._init_vals(init_pdf=init_pdf,
                        init_density=init_density,
                        init_vel=init_vel)

        """ pdf for boundary handling """
        self._pdf_pre = np.zeros_like(self.pdf)
        self._pdf_mid = np.zeros_like(self.pdf)

        assert 0 < omega < 2
        self._omega = omega

    @property
    def pdf(self) -> np.ndarray:
        return self._pdf

    @pdf.setter
    def pdf(self) -> None:
        raise NotImplementedError("pdf is not supposed to change from outside.")

    @property
    def pdf_pre(self) -> np.ndarray:
        return self._pdf_pre

    @pdf_pre.setter
    def pdf_pre(self) -> None:
        raise NotImplementedError("pdf_pre is not supposed to change from outside.")

    @property
    def pdf_mid(self) -> np.ndarray:
        return self._pdf_mid

    @pdf_mid.setter
    def pdf_mid(self) -> None:
        raise NotImplementedError("pdf_mid is not supposed to change from outside.")

    @property
    def pdf_eq(self) -> np.ndarray:
        return self._pdf_eq

    @pdf_eq.setter
    def pdf_eq(self) -> None:
        raise NotImplementedError("pdf_eq is not supposed to change from outside.")

    @property
    def density(self) -> np.ndarray:
        return self._density

    @density.setter
    def density(self) -> None:
        raise NotImplementedError("density is not supposed to change from outside.")

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self) -> None:
        raise NotImplementedError("velocity is not supposed to change from outside.")

    @property
    def lattice_grid_shape(self) -> Tuple[int, int]:
        return self._lattice_grid_shape

    @property
    def omega(self) -> float:
        return self._omega

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

    def _init_vals(self, init_pdf: Optional[np.ndarray] = None,
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
        self._density = np.array(np.sum(self.pdf, axis=-1))

    def update_velocity(self) -> None:
        assert self._finish_initialize

        self._velocity[:, :, 0] = (
            np.sum(self.pdf[:, :, AdjacentAttributes.x_right], axis=-1)
            - np.sum(self.pdf[:, :, AdjacentAttributes.x_left], axis=-1)
        ) / np.maximum(self.density, EPS)

        self._velocity[:, :, 1] = (
            np.sum(self.pdf[:, :, AdjacentAttributes.y_upper], axis=-1)
            - np.sum(self.pdf[:, :, AdjacentAttributes.y_lower], axis=-1)
        ) / np.maximum(self.density, EPS)

    def update_pdf(self) -> None:
        """Update the current pdf based on the streaming operator """
        vs = AdjacentAttributes.velocity_direction_set

        next_pdf = np.zeros_like(self.pdf)
        for i in range(9):
            next_pdf[..., i] = np.roll(self.pdf[..., i], vs[i], axis=(0, 1))

        self._pdf = next_pdf

    def overwrite_pdf(self, new_pdf: np.ndarray) -> None:
        assert self.pdf.shape == new_pdf.shape
        self._pdf = new_pdf

    def _apply_local_equilibrium(self) -> None:
        self._pdf_eq = local_equilibrium(velocity=self.velocity, density=self.density)

    def local_equilibrium_pdf_update(self) -> None:
        self._apply_local_equilibrium()
        self._pdf = deepcopy(self._pdf_eq)    

    def lattice_boltzmann_step(
        self,
        boundary_handling: Optional[Callable[['FluidField2D'], None]] = None
    ) -> None:

        self._apply_local_equilibrium()

        self._pdf_pre = deepcopy(self.pdf)
        self._pdf_mid = (self.pdf + (self.pdf_eq - self.pdf) * self._omega)
        self._pdf = deepcopy(self.pdf_mid)
        self.update_pdf()

        if boundary_handling is not None:
            boundary_handling(self)

        self.update_density()
        self.update_velocity()
