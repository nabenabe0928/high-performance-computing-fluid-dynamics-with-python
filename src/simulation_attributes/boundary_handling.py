from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from src.simulation_attributes.formula import AdjacentIndices, FluidField2D


class AbstractBoundaryHandling(object, metaclass=ABCMeta):
    @abstractmethod
    def boundary_handling(self, field: FluidField2D, **kwargs) -> np.ndarray:
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
    def __init__(self, field: FluidField2D, **kwargs: Dict[str, Any]):
        self._boundary = np.zeros(field.lattice_grid_shape, np.bool8)
        self._reflected_direction = AdjacentIndices.reflected_direction()
        self._boundary_indices = np.zeros(9).astype(np.bool8)
        self._finish_initialize = False

    @property
    def boundary(self) -> np.ndarray:
        return self._boundary

    @property
    def boundary_indices(self) -> np.ndarray:
        return self._boundary_indices

    @property
    def reflected_direction(self) -> None:
        return self._reflected_direction

    def _init_boundary_indices(self) -> None:
        """
        Suppose walls are not disjointed and do not have curves
        and they exist only at the edges of the field.
        """
        assert self._finish_initialize
        if np.all(self.boundary[0, :]):
            self._boundary_indices[[1, 5, 8]] = True
        if np.all(self.boundary[-1, :]):
            self._boundary_indices[[3, 6, 7]] = True
        if np.all(self.boundary[:, 0]):
            self._boundary_indices[[4, 7, 8]] = True
        if np.all(self.boundary[:, -1]):
            self._boundary_indices[[2, 5, 6]] = True

        self._reflected_direction = self.reflected_direction[self._boundary_indices]

    def init_boundary(self, init_boundary: np.ndarray) -> None:
        assert not self._finish_initialize
        assert self.boundary.shape == init_boundary.shape
        self._boundary = init_boundary.astype(np.bool8)
        self._finish_initialize = True
        self._init_boundary_indices()


class RigidWall(BaseWall, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D):
        super().__init__(field)

    def boundary_handling(self, field: FluidField2D) -> None:
        pdf_post = deepcopy(field.pdf)

        pdf_post[self.boundary, self.reflected_direction] = \
            field.pdf_pre[self.boundary, self.boundary_indices]

        field.overwrite_pdf(new_pdf=pdf_post)


class MovingWall(BaseWall, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, wall_vel: np.ndarray):
        super().__init__(field)
        self.weights = AdjacentIndices.weights()
        self.vs = AdjacentIndices.velocity_direction_set()
        self.vs_dot_wall_vel3 = np.array([])
        self._wall_vel = wall_vel  # shape (2, )
        self._finish_precompute = False

    @property
    def wall_vel(self) -> np.ndarray:
        """ The velocity of the wall """
        return self._wall_vel

    def _precompute(self):
        assert not self._finish_precompute
        self.weights = self.weights[self.boundary_indices]
        self.vs = self.vs[self.boundary_indices]
        self.vs_dot_wall_vel3 = 3 * (self.vs @ self.wall_vel)
        self._finish_precompute = True

    def boundary_handling(self, field: FluidField2D) -> None:
        if not self._finish_precompute:
            self._precompute()

        pdf_post = deepcopy(field.pdf)
        pdf_post[self.boundary, self.reflected_direction] = (
            field.pdf_pre[self.boundary, self.boundary_indices]
            - 2 * self.weights * field.density * self.vs_dot_wall_vel3
        )

        field.overwrite_pdf(new_pdf=pdf_post)
