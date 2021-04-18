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


class BaseWall():
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray, **kwargs: Dict[str, Any]):
        """
        Attributes:
            _boundary (np.ndarray):
                If there are walls (or boundary) or not.
                Each element is True or False
                The shape is (X, Y, 9).

            _boundary_indices (np.ndarray):
                It stands for which directions (from 9 adjacent cells) can have the boundary.
                The shape is (n_direction, ) where n_direction is smaller than 9.

            _reflected_indices (np.ndarray):
                The corresponding indices for the bouncing direction of _boundary_indices.
                The shape is (n_direction, ).
        """
        self._boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._reflected_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._reflected_indices = AdjacentAttributes.reflected_direction
        self._boundary_indices = np.arange(9)
        self._finish_initialize = False

        self._init_boundary(init_boundary)

    @property
    def boundary(self) -> np.ndarray:
        return self._boundary

    @boundary.setter
    def boundary(self) -> None:
        raise NotImplementedError("boundary is not supposed to change from outside.")

    @property
    def reflected_boundary(self) -> np.ndarray:
        return self._reflected_boundary

    @reflected_boundary.setter
    def reflected_boundary(self) -> None:
        raise NotImplementedError("reflected_boundary is not supposed to change from outside.")

    @property
    def boundary_indices(self) -> np.ndarray:
        return self._boundary_indices

    @boundary_indices.setter
    def boundary_indices(self) -> None:
        raise NotImplementedError("boundary_indices is not supposed to change from outside.")

    @property
    def reflected_indices(self) -> np.ndarray:
        return self._reflected_indices

    @reflected_indices.setter
    def reflected_indices(self) -> None:
        raise NotImplementedError("reflected_indices is not supposed to change from outside.")

    def _init_boundary_indices(self, init_boundary: np.ndarray) -> None:
        """
        Suppose walls are not disjointed and do not have curves
        and they exist only at the edges of the field.
        """
        assert not self._finish_initialize
        if np.all(init_boundary[0, :]):
            self._boundary_indices = np.array([1, 5, 8])
        if np.all(init_boundary[-1, :]):
            self._boundary_indices = np.array([3, 6, 7])
        if np.all(init_boundary[:, 0]):
            self._boundary_indices = np.array([4, 7, 8])
        if np.all(init_boundary[:, -1]):
            self._boundary_indices = np.array([2, 5, 6])

        self._reflected_indices = self.reflected_indices[self.boundary_indices]

    def _init_boundary(self, init_boundary: np.ndarray) -> None:
        assert not self._finish_initialize
        assert self.boundary.shape[:-1] == init_boundary.shape
        init_boundary = init_boundary.astype(np.bool8)
        self._init_boundary_indices(init_boundary)

        for bidx, ridx in zip(self.boundary_indices, self.reflected_indices):
            self._boundary[:, :, bidx] = init_boundary
            self._reflected_boundary[:, :, ridx] = init_boundary

        self._finish_initialize = True


class RigidWall(BaseWall, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray):
        super().__init__(field, init_boundary)

    def boundary_handling(self, field: FluidField2D) -> None:
        pdf_post = deepcopy(field.pdf)
        pdf_post[self.reflected_boundary] = field.pdf_pre[self.boundary]
        field.overwrite_pdf(new_pdf=pdf_post)


class MovingWall(BaseWall, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, init_boundary: np.ndarray, wall_vel: np.ndarray):
        """
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
        ws = AdjacentAttributes.weights[self.boundary_indices]
        vs = AdjacentAttributes.velocity_direction_set[self.boundary_indices]

        self._weighted_vel_dot_wall_vel6 = np.zeros_like(self.boundary, np.float32)
        for bidx, v, w in zip(self.boundary_indices, vs, ws):
            self._weighted_vel_dot_wall_vel6[:, :, bidx] = 6 * w * (v @ self.wall_vel)

        self._finish_precompute = True

    def boundary_handling(self, field: FluidField2D) -> None:
        if not self._finish_precompute:
            self._precompute()

        pdf_post = deepcopy(field.pdf)
        average_density = field.density.mean()

        pdf_post[self.reflected_boundary] = (
            field.pdf_pre[self.boundary]
            - average_density *
            self.weighted_vel_dot_wall_vel6[self.boundary]
        )

        field.overwrite_pdf(new_pdf=pdf_post)
