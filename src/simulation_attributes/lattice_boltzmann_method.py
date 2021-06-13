import numpy as np
from tqdm import trange
from typing import Callable, Optional, Tuple
from copy import deepcopy

from src.utils.constants import AdjacentAttributes
from src.utils.parallel_computation import ChunkedGridManager
from src.utils.utils import make_directories_to_path, omega2viscosity


EPS = 1e-12
BoundaryHandlingFuncType = Callable[['LatticeBoltzmannMethod'], None]
ProcessFuncType = Callable[['LatticeBoltzmannMethod', int], None]


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
                 grid_manager: Optional[ChunkedGridManager] = None,
                 dir_name: Optional[str] = None):
        """
        This class computes and stores the density and velocity field
        given (x, y) and initializes them at given v and t = 0.
        We update each value every time we update().

        Attributes:
            _finish_initialize (bool):
                This is used to avoid multiple initialization.

            grid_manager (Optional[ChunkedGridManager]):
                The process manager for the parallel computation settings.
                It allows this instance to know which location
                the current process lies and which process it should communicate.

            local_density_sum (float):
                Local density sum for the computation of global density average
                in the parallel settings.

            global_density_average (float):
                Global density average is computed in the process with the rank of 0
                and each process receives this value from the process.

            recvbuf (List[np.ndarray]):
                The array to receive arrays from other processes.
                Since the communication direction are three types:
                `x direction`, `y direction`, `diagonal direction`,
                we use buffer with three types of shapes.
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
        self.dir_name = dir_name

        self._init_vals(init_pdf=init_pdf,
                        init_density=init_density,
                        init_vel=init_vel)

        """ pdf for boundary handling """
        self._pdf_pre = np.zeros_like(self.pdf)

        assert 0 < omega < 2
        self._omega = omega
        self._viscosity = omega2viscosity(omega)
        self.local_density_sum = 0.0
        self.global_density_average = 0.0
        self.recvbuf = [
            np.zeros_like(self.pdf_pre[:, 0, ...]),
            np.zeros_like(self.pdf_pre[0, ...]),
            np.zeros_like(self.pdf_pre[0, 0, ...])
        ]

    def __call__(
        self,
        total_time_steps: int,
        proc: Optional[ProcessFuncType] = None,
        boundary_handling: Optional[BoundaryHandlingFuncType] = None
    ) -> None:

        self.local_equilibrium_pdf_update()
        for t in trange(total_time_steps + 1):
            self.lattice_boltzmann_step(boundary_handling=boundary_handling)
            if proc is not None:
                proc(self, t)

    @property
    def pdf(self) -> np.ndarray:
        """
        Returns:
            _pdf (np.ndarray):
                probability density function of the 8 adjacent cells
                and the target cell. The shape is (X, Y, 9).
        """
        return self._pdf

    @pdf.setter
    def pdf(self) -> None:
        raise NotImplementedError("pdf is not supposed to change from outside.")

    @property
    def pdf_pre(self) -> np.ndarray:
        """
        Returns:
            _pdf_pre (np.ndarray):
                linear interpolation of _pdf and _pdf_eq.
                It is used in the boundary handling.
                The shape is (X, Y, 9).
        """
        return self._pdf_pre

    @pdf_pre.setter
    def pdf_pre(self) -> None:
        raise NotImplementedError("pdf_pre is not supposed to change from outside.")

    @property
    def pdf_eq(self) -> np.ndarray:
        """
        Returns:
            _pdf_eq (np.ndarray):
                probability density function after the local equilibrium.
                The shape is (X, Y, 9).
        """
        return self._pdf_eq

    @pdf_eq.setter
    def pdf_eq(self) -> None:
        raise NotImplementedError("pdf_eq is not supposed to change from outside.")

    @property
    def density(self) -> np.ndarray:
        """
        Returns:
            _density (np.ndarray):
                density of the given location (x, y).
                The shape is (X, Y)
        """
        return self._density

    @density.setter
    def density(self) -> None:
        raise NotImplementedError("density is not supposed to change from outside.")

    @property
    def velocity(self) -> np.ndarray:
        """
        Returns:
            _velocity (np.ndarray):
                velocity of the given location (x, y).
                The shape is (X, Y, 2).
        """
        return self._velocity

    @velocity.setter
    def velocity(self) -> None:
        raise NotImplementedError("velocity is not supposed to change from outside.")

    @property
    def lattice_grid_shape(self) -> Tuple[int, int]:
        """
        Returns:
            lattice_grid_shape (Tuple[int, int]) = (X, Y):
                The shape of lattice grid
                X (int): The size of x axis
                Y (int): The size of y axis
        """
        return self._lattice_grid_shape

    @property
    def omega(self) -> float:
        """
        Returns:
            _omega (float): relaxation term.
        """
        return self._omega

    @property
    def viscosity(self) -> float:
        """
        Returns:
            _viscosity (float): viscosity of the fluid.
        """
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
            np.sum(self.pdf[:, :, AdjacentAttributes.y_top], axis=-1)
            - np.sum(self.pdf[:, :, AdjacentAttributes.y_bottom], axis=-1)
        ) / np.maximum(self.density, EPS)

    def update_pdf(self) -> None:
        """ Update the current pdf based on the streaming operator """
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
            self._communicate_for_pdf()
            self._communicate_for_density()
            # Wait for all the communications
            self.grid_manager.comm.Barrier()
        else:
            self.global_density_average = float(self.density.mean())

        self._pdf = deepcopy(self.pdf_pre)
        self.update_pdf()

        if boundary_handling is not None:
            """ use pdf, pdf_pre, density, pdf_eq, velocity inside """
            boundary_handling(self)

        self.update_density()
        self.update_velocity()

    def _communicate_for_pdf(self) -> None:
        """ Communicate the pdf_pre with neighbors """
        step_to_idx = self.grid_manager._step_to_idx
        for dir in self.grid_manager.neighbor_directions:
            dx, dy = AdjacentAttributes.velocity_direction_set[dir]
            sendidx, recvidx = step_to_idx(dx, dy, True), step_to_idx(dx, dy, False)
            neighbor = self.grid_manager.get_neighbor_rank(dir)

            if dx == 0:  # communication for x direction
                sendbuf = self.pdf_pre[:, sendidx, ...].copy()
                self.grid_manager.rank_grid.Sendrecv(sendbuf=sendbuf, dest=neighbor,
                                                     recvbuf=self.recvbuf[0], source=neighbor)
                self.pdf_pre[:, recvidx, ...] = self.recvbuf[0]
            elif dy == 0:  # communication for y direction
                sendbuf = self.pdf_pre[sendidx, ...].copy()
                self.grid_manager.rank_grid.Sendrecv(sendbuf=sendbuf, dest=neighbor,
                                                     recvbuf=self.recvbuf[1], source=neighbor)
                self.pdf_pre[recvidx, ...] = self.recvbuf[1]
            else:  # communication for diagonal direction
                sendbuf = self.pdf_pre[sendidx[0], sendidx[1], ...].copy()
                self.grid_manager.rank_grid.Sendrecv(sendbuf=sendbuf, dest=neighbor,
                                                     recvbuf=self.recvbuf[2], source=neighbor)
                self.pdf_pre[recvidx[0], recvidx[1], ...] = self.recvbuf[2]

    def _communicate_for_density(self) -> None:
        x_start, x_end = self.grid_manager.x_valid_slice
        y_start, y_end = self.grid_manager.y_valid_slice

        self.local_density_sum = self.density[x_start:x_end, y_start:y_end].sum()
        self.global_density_average = 0.0

        sendbuf = np.ones(1, dtype=np.float64) * self.local_density_sum
        recvbuf = None
        if self.grid_manager.rank == 0:  # compute average in the process of the rank 0
            recvbuf = np.empty([self.grid_manager.size, 1], dtype=np.float64)

        # gather local sums in the process of the rank 0
        self.grid_manager.comm.Gather(sendbuf, recvbuf, root=0)

        sendbuf = None
        n_grids = self.grid_manager.global_grid_size[0] * self.grid_manager.global_grid_size[1]

        if self.grid_manager.rank == 0:
            assert recvbuf is not None
            # compute the global average in the process of the rank 0
            self.global_density_average = recvbuf.sum() / n_grids
            sendbuf = np.ones([self.grid_manager.size, 1], dtype=np.float64) * self.global_density_average

        recvbuf = np.empty(1, dtype=np.float64)
        self.grid_manager.comm.Scatter(sendbuf, recvbuf, root=0)

        self.global_density_average = recvbuf[0]

    def save_velocity_field(self, t: int) -> None:
        """
        The numpy array save for MPI settings.

        Args:
            t (int): the current time step.
        """
        assert self.grid_manager is not None

        x_start, x_end = self.grid_manager.x_valid_slice
        y_start, y_end = self.grid_manager.y_valid_slice

        path = f'log/{self.dir_name}/npy/'
        make_directories_to_path(path)
        abs_file_name = f'{path}v_abs{t:0>6}.npy'
        x_file_name = f'{path}v_x{t:0>6}.npy'
        y_file_name = f'{path}v_y{t:0>6}.npy'

        self.grid_manager.save_mpiio(
            abs_file_name,
            np.linalg.norm(self.velocity[x_start:x_end, y_start:y_end], axis=-1)
        )
        self.grid_manager.save_mpiio(x_file_name, self.velocity[x_start:x_end, y_start:y_end, 0])
        self.grid_manager.save_mpiio(y_file_name, self.velocity[x_start:x_end, y_start:y_end, 1])
