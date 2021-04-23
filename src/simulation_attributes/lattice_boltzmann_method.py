import numpy as np
from typing import Callable, Optional, Tuple
from copy import deepcopy

# from src.utils.boundary_handling import BaseBoundary
from src.utils.constants import (
    AdjacentAttributes,
    DirectionIndicators,
    DIRECTION2VEC
)
from src.utils.parallel_computation import ChunkedGridManager


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


class LatticeBoltzmannMethod():
    def __init__(self, X: int, Y: int, omega: float = 0.5,
                 init_pdf: Optional[np.ndarray] = None,
                 init_density: Optional[np.ndarray] = None,
                 init_vel: Optional[np.ndarray] = None,
                 grid_manager: Optional[ChunkedGridManager] = None):
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
        if grid_manager is not None:  # add ghost cells
            X, Y = grid_manager.buffer_grid_size

        self._pdf = np.zeros((X, Y, 9))
        self._pdf_eq = np.zeros((X, Y, 9))
        self._density = np.zeros((X, Y))
        self._velocity = np.zeros((X, Y, 2))
        self._lattice_grid_shape = (X, Y)
        self._finish_initialize = False
        self.grid_manager = grid_manager

        self._init_vals(init_pdf=init_pdf,
                        init_density=init_density,
                        init_vel=init_vel)

        """ pdf for boundary handling """
        self._pdf_pre = np.zeros_like(self.pdf)

        assert 0 < omega < 2
        self._omega = omega
        self._viscosity = 1. / 3. * (1. / omega - 0.5)
        """ TODO: make the function for them """
        self.local_density_sum = 0.0
        self.global_density_average = 0.0

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

    @property
    def viscosity(self) -> float:
        return self._viscosity

    def is_parallel(self) -> bool:
        return self.grid_manager is not None

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
        assert self._finish_initialize

        vs = AdjacentAttributes.velocity_direction_set

        next_pdf = np.zeros_like(self.pdf)
        for i in range(9):
            """ for axis j, shift vs[i][j] """
            next_pdf[..., i] = np.roll(self.pdf[..., i], shift=vs[i], axis=(0, 1))

        self._pdf = next_pdf

    def _apply_local_equilibrium(self) -> None:
        self._pdf_eq = local_equilibrium(velocity=self.velocity, density=self.density)

    def local_equilibrium_pdf_update(self) -> None:
        self._apply_local_equilibrium()
        self._pdf = deepcopy(self._pdf_eq)

    def lattice_boltzmann_step(
        self,
        boundary_handling: Optional[Callable[['LatticeBoltzmannMethod'], None]] = None
    ) -> None:

        self._apply_local_equilibrium()

        self._pdf_pre = (self.pdf + (self.pdf_eq - self.pdf) * self._omega)

        if self.is_parallel():
            # TODO: average density computation
            self._communicate_for_pdf()
            self._communicate_for_density()
            self.grid_manager.comm.Barrier()
        else:
            self.global_density_average = self.density.mean()

        self._pdf = deepcopy(self.pdf_pre)
        self.update_pdf()

        if boundary_handling is not None:
            """ use pdf, pdf_pre, density, pdf_eq, velocity inside """
            boundary_handling(self)

        self.update_density()
        self.update_velocity()

    def _communicate_for_pdf(self) -> None:
        """TODO: pdf_pre and check corner points"""
        step_to_idx = self.grid_manager._step_to_idx
        for dir in DirectionIndicators:
            dx, dy = DIRECTION2VEC[dir]
            sendidx, recvidx = step_to_idx(dx, dy, True), step_to_idx(dx, dy, False)

            if self.grid_manager.exist_neighbor(dir):
                neighbor = self.grid_manager.get_neighbor_rank(dir)

                if dx == 0:
                    sendbuf = self.pdf_pre[:, sendidx, ...].copy()
                    recvbuf = np.zeros_like(self.pdf_pre[:, recvidx, ...])
                    self.grid_manager.rank_grid.Sendrecv(sendbuf=sendbuf, dest=neighbor,
                                                         recvbuf=recvbuf, source=neighbor)
                    self.pdf_pre[:, recvidx, ...] = recvbuf
                elif dy == 0:
                    sendbuf = self.pdf_pre[sendidx, ...].copy()
                    recvbuf = np.zeros_like(self.pdf_pre[recvidx, ...])
                    self.grid_manager.rank_grid.Sendrecv(sendbuf=sendbuf, dest=neighbor,
                                                         recvbuf=recvbuf, source=neighbor)
                    self.pdf_pre[recvidx, ...] = recvbuf
                else:
                    sendbuf = self.pdf_pre[sendidx[0], sendidx[1], ...].copy()
                    recvbuf = np.zeros_like(self.pdf_pre[recvidx[0], recvidx[1], ...])
                    self.grid_manager.rank_grid.Sendrecv(sendbuf=sendbuf, dest=neighbor,
                                                         recvbuf=recvbuf, source=neighbor)
                    self.pdf_pre[recvidx[0], recvidx[1], ...] = recvbuf

    def _communicate_for_density(self) -> None:
        x_start, x_end = self.grid_manager.x_local_slice
        y_start, y_end = self.grid_manager.y_local_slice

        self.local_density_sum = self.density[x_start:x_end, y_start:y_end].sum()
        self.global_density_average = 0.0

        sendbuf = np.ones(1, dtype=np.float64) * self.local_density_sum
        recvbuf = None
        if self.grid_manager.rank == 0:
            recvbuf = np.empty([self.grid_manager.size, 1], dtype=np.float64)

        self.grid_manager.comm.Gather(sendbuf, recvbuf, root=0)

        sendbuf = None
        n_grids = self.grid_manager.global_grid_size[0] * self.grid_manager.global_grid_size[1]

        if self.grid_manager.rank == 0:
            assert recvbuf is not None
            self.global_density_average = recvbuf.sum() / n_grids
            sendbuf = np.ones([self.grid_manager.size, 1], dtype=np.float64) * self.global_density_average

        recvbuf = np.empty(1, dtype=np.float64)
        self.grid_manager.comm.Scatter(sendbuf, recvbuf, root=0)

        self.global_density_average = recvbuf[0]
