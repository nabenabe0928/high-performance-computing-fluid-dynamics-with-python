from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from src.simulation_attributes.formula import AdjacentAttributes, FluidField2D


class AbstractBoundaryHandling(object, metaclass=ABCMeta):
    @abstractmethod
    def boundary_handling(self, field: FluidField2D) -> None:
        """
        Compute the PDF using pdf_pre, pdf_mid, pdf and density, velocity
        and return the PDF after boundary handling.

        Args:
            field (FluidField2D)

        Returns:
            pdf_post (np.ndarray):
                The pdf after the boundary handling.
                The shape is (X, Y, 9).
        """
        raise NotImplementedError

    @abstractmethod
    def _precompute(self) -> None:
        """ pre-computation if required """
        raise NotImplementedError


class BaseBoundary():
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray,
                 pressure_variation: bool = False, **kwargs: Dict[str, Any]):
        """
        Attributes:
            _out_boundary (np.ndarray):
                Direction to come out.
                In other words, if there are walls (or boundary) or not
                or the outlet of pipes.
                Each element is True or False with the shape is (X, Y, 9).

            _in_boundary (np.ndarray):
                Direction to come in.
                In other words, if the reflected direction of _out_boundary
                or the inlet of pipes.
                Each element is True or False with the shape is (X, Y, 9).

            _out_indices (np.ndarray):
                It stands for which directions (from 9 adjacent cells) can have the out-boundary.
                The shape is (n_direction, ) where n_direction is smaller than 9.

            _in_indices (np.ndarray):
                The corresponding indices for the bouncing direction of _out_indices.
                The shape is (n_direction, ).
        """
        self._out_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._out_indices = np.arange(9)
        self._in_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._in_indices = AdjacentAttributes.reflected_direction
        self._finish_initialize = False

        self._init_boundary(init_boundary, pressure_variation=pressure_variation)

    @property
    def in_boundary(self) -> np.ndarray:
        return self._in_boundary

    @in_boundary.setter
    def in_boundary(self) -> None:
        raise NotImplementedError("in_boundary is not supposed to change from outside.")

    @property
    def out_boundary(self) -> np.ndarray:
        return self._out_boundary

    @out_boundary.setter
    def out_boundary(self) -> None:
        raise NotImplementedError("out_boundary is not supposed to change from outside.")

    @property
    def in_indices(self) -> np.ndarray:
        return self._in_indices

    @in_indices.setter
    def in_indices(self) -> None:
        raise NotImplementedError("in_indices is not supposed to change from outside.")

    @property
    def out_indices(self) -> np.ndarray:
        return self._out_indices

    @out_indices.setter
    def out_indices(self) -> None:
        raise NotImplementedError("out_indices is not supposed to change from outside.")

    def _init_boundary_indices(self, init_boundary: np.ndarray, pressure_variation: bool) -> None:
        """
        Suppose walls are not disjointed and do not have curves
        and they exist only at the edges of the field.

        Args:
            init_boundary (np.ndarray):
                The True or False array with the shape of (X, Y).
                If True, there is a boundary.

            pressure_variation (bool):
                If True, pressure variation dominates the influence from
                the collision with the wall.
        """
        assert not self._finish_initialize
        out_indices = []
        if not pressure_variation:
            if np.all(init_boundary[0, :]):
                out_indices += [1, 5, 8]
            if np.all(init_boundary[-1, :]):
                out_indices += [3, 6, 7]
            if np.all(init_boundary[:, 0]):
                out_indices += [4, 7, 8]
            if np.all(init_boundary[:, -1]):
                out_indices += [2, 5, 6]
        else:
            horiz = (np.all(init_boundary[0, :]) and np.all(init_boundary[-1, :]))
            vert = np.all(init_boundary[:, 0]) and np.all(init_boundary[:, -1])
            assert vert or horiz
            if horiz:
                # left to right
                out_indices = [1, 5, 8]
            else:
                # bottom to top
                out_indices = [2, 5, 6]

        self._out_indices = np.array(out_indices)
        self._in_indices = self.in_indices[self.out_indices]

    def _init_boundary(self, init_boundary: np.ndarray, pressure_variation: bool) -> None:
        assert not self._finish_initialize
        assert self.out_boundary.shape[:-1] == init_boundary.shape
        init_boundary = init_boundary.astype(np.bool8)
        self._init_boundary_indices(init_boundary, pressure_variation)

        for out_idx, in_idx in zip(self.out_indices, self.in_indices):
            self._out_boundary[:, :, out_idx] = init_boundary
            self._in_boundary[:, :, in_idx] = init_boundary

        self._finish_initialize = True


class RigidWall(BaseBoundary, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray):
        super().__init__(field, init_boundary)

    def _precompute(self) -> None:
        pass

    def boundary_handling(self, field: FluidField2D) -> None:
        pdf_post = deepcopy(field.pdf)
        pdf_post[self.in_boundary] = field.pdf_pre[self.out_boundary]
        field.overwrite_pdf(new_pdf=pdf_post)


class MovingWall(BaseBoundary, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray, wall_vel: np.ndarray):
        """
        Attributes:
            _wall_vel (np.ndarray):
                The velocity vector of the movement of the wall

            _weighted_vel_dot_wall_vel6 (np.ndarray):
                The computation results of
                2 * wi * rhow * (ci @ uw) / cs ** 2
                in the equation for the moving wall.

        Example:
            ##### -> the wall will move to this direction.
             ...
             ...
        """
        super().__init__(field, init_boundary)
        self._weighted_vel_dot_wall_vel6 = np.array([])
        self._wall_vel = wall_vel  # shape (2, )
        self._finish_precompute = False

    @property
    def wall_vel(self) -> np.ndarray:
        """ The velocity of the wall """
        return self._wall_vel

    @wall_vel.setter
    def wall_vel(self) -> None:
        """ The velocity of the wall """
        raise NotImplementedError("wall_vel is not supposed to change from outside.")

    @property
    def weighted_vel_dot_wall_vel6(self) -> np.ndarray:
        """
        The inner product of each velocity set and wall velocity multiplied by
        the constant value 6 and weights for the corresponding adjacent cell.
        """
        return self._weighted_vel_dot_wall_vel6

    @weighted_vel_dot_wall_vel6.setter
    def weighted_vel_dot_wall_vel6(self) -> None:
        """ The velocity of the wall """
        raise NotImplementedError("weighted_vel_dot_wall_vel6 is not supposed to change from outside.")

    def _precompute(self) -> None:
        assert not self._finish_precompute
        ws = AdjacentAttributes.weights[self.out_indices]
        vs = AdjacentAttributes.velocity_direction_set[self.out_indices]

        self._weighted_vel_dot_wall_vel6 = np.zeros_like(self.out_boundary, np.float32)
        for out_idx, v, w in zip(self.out_indices, vs, ws):
            self._weighted_vel_dot_wall_vel6[:, :, out_idx] = 6 * w * (v @ self.wall_vel)

        self._finish_precompute = True

    def boundary_handling(self, field: FluidField2D) -> None:
        if not self._finish_precompute:
            self._precompute()

        pdf_post = deepcopy(field.pdf)
        average_density = field.density.mean()

        pdf_post[self.in_boundary] = (
            field.pdf_pre[self.out_boundary]
            - average_density *
            self.weighted_vel_dot_wall_vel6[self.out_boundary]
        )

        field.overwrite_pdf(new_pdf=pdf_post)


class PeriodicBoundaryConditions(BaseBoundary, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray,
                 in_density_factor: float, out_density_factor: float):

        # left to right
        horiz = (np.all(init_boundary[0, :]) and np.all(init_boundary[-1, :]))
        # bottom to top
        vert = np.all(init_boundary[:, 0]) and np.all(init_boundary[:, -1])
        assert vert or horiz
        X, Y = field.lattice_grid_shape

        super().__init__(field, init_boundary, pressure_variation=True)
        boundary_shape = X if horiz else Y
        self._in_density = 3 * in_density_factor * np.ones(boundary_shape)
        self._out_density = 3 * out_density_factor * np.ones(boundary_shape)

    @property
    def in_density(self) -> float:
        return self._in_density

    @in_density.setter
    def in_density(self) -> None:
        raise NotImplementedError("in_density is not supposed to change from outside.")

    @property
    def out_density(self) -> float:
        return self._out_density

    @out_density.setter
    def out_density(self) -> None:
        raise NotImplementedError("out_density is not supposed to change from outside.")

    def boundary_handling(self, field: FluidField2D) -> None:
        pdf_eq = field.pdf_eq
        pdf_post = deepcopy(field.pdf)
        pdf_post[self.in_boundary] = field.pdf_pre[self.out_boundary]
        field.overwrite_pdf(new_pdf=pdf_post)


"""
def periodic_with_pressure_variations(boundary: np.ndarray, p_in: float, p_out: float) \
        -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:

    def bc(f_pre_streaming: np.ndarray, density: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        assert boundary.shape == f_pre_streaming.shape[0:2]

        f_eq = equilibrium_distr_func(density, velocity)
        f_eq_in = equilibrium_distr_func(density_in, velocity[-2, ...]).squeeze()
        f_pre_streaming[0, ..., change_directions_1] = f_eq_in[..., change_directions_1].T + (
                f_pre_streaming[-2, ..., change_directions_1] - f_eq[-2, ..., change_directions_1])

        f_eq_out = equilibrium_distr_func(density_out, velocity[1, ...]).squeeze()
        f_pre_streaming[-1, ..., change_directions_2] = f_eq_out[..., change_directions_2].T + (
                f_pre_streaming[1, ..., change_directions_2] - f_eq[1, ..., change_directions_2])

        return f_pre_streaming

    return bc
"""
