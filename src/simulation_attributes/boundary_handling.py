from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List
from enum import IntEnum

import numpy as np

from src.simulation_attributes.formula import AdjacentAttributes, FluidField2D, local_equilibrium


class DirectionIndicators(IntEnum):
    RIGHT: int = 0
    LEFT: int = 1
    TOP: int = 2
    BOTTOM: int = 3


def get_direction_representor(boundary: np.ndarray) -> str:
    indices = [i for i, b in enumerate(boundary) if b]
    """
    Adjacent cell indices
    6 2 5
    3 0 1
    7 4 8
    """
    if indices == [3, 6, 7]:
        return "<"
    if indices == [1, 5, 8]:
        return ">"
    if indices == [2, 5, 6]:
        return "^"
    if indices == [4, 7, 8]:
        return "v"
    else:
        return "*"


class AbstractBoundaryHandling(object, metaclass=ABCMeta):
    @abstractmethod
    def boundary_handling(self, field: FluidField2D) -> None:
        """
        Compute the PDF using pdf_pre, pdf_mid, pdf and density, velocity
        and return the PDF after boundary handling.
        In order to be able to joint multiple boundary handlings,
        the update of the PDF has to be performed inside the function.

        Args:
            field (FluidField2D)
        """
        raise NotImplementedError


class BaseBoundary():
    def __init__(self, field: FluidField2D, boundary_locations: List[DirectionIndicators],
                 pressure_variation: bool = False, visualize_wall: bool = False,
                 **kwargs: Dict[str, Any]):
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
        X, Y = field.lattice_grid_shape
        self._out_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._out_indices = np.arange(9)
        self._in_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._in_indices = AdjacentAttributes.reflected_direction
        self._finish_initialize = False
        self._lattice_grid_shape = field.lattice_grid_shape
        self._boundary_locations = boundary_locations
        self._visualize_wall = visualize_wall

        self._init_boundary(pressure_variation=pressure_variation)

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

    @property
    def boundary_locations(self) -> List[DirectionIndicators]:
        return self._boundary_locations

    @boundary_locations.setter
    def boundary_locations(self) -> None:
        raise NotImplementedError("boundary_locations is not supposed to change from outside.")

    def _init_boundary_indices(self, pressure_variation: bool) -> None:
        """
        Suppose walls are not disjointed and do not have curves
        and they exist only at the edges of the field.

        Args:
            pressure_variation (bool):
                If True, pressure variation dominates the influence from
                the collision with the wall.
        """
        assert not self._finish_initialize
        out_indices = []
        if not pressure_variation:
            if DirectionIndicators.LEFT in self.boundary_locations:
                out_indices += [3, 6, 7]
            if DirectionIndicators.RIGHT in self.boundary_locations:
                out_indices += [1, 5, 8]
            if DirectionIndicators.BOTTOM in self.boundary_locations:
                out_indices += [4, 7, 8]
            if DirectionIndicators.TOP in self.boundary_locations:
                out_indices += [2, 5, 6]
        else:
            # left to right (the flow of particles)
            horiz = (DirectionIndicators.LEFT in self.boundary_locations and
                     DirectionIndicators.RIGHT in self.boundary_locations)
            # bottom to top
            vert = (DirectionIndicators.TOP in self.boundary_locations and
                    DirectionIndicators.BOTTOM in self.boundary_locations)
            assert vert or horiz
            if horiz:
                # left to right
                out_indices = [1, 5, 8]
            else:
                # bottom to top
                out_indices = [2, 5, 6]

        self._out_indices = np.array(out_indices)
        self._in_indices = self.in_indices[self.out_indices]

    def _allocate_boundary_conditions(self, in_idx: int, out_idx: int) -> None:
        """
        Adjacent cell indices
        6 2 5
        3 0 1
        7 4 8
        """
        if (
            DirectionIndicators.LEFT in self.boundary_locations and
            (out_idx in [3, 6, 7] and in_idx in [1, 5, 8])  # Wall exists left
        ):
            self._out_boundary[0, :, out_idx] = True
            self._in_boundary[0, :, in_idx] = True
        if (
            DirectionIndicators.RIGHT in self.boundary_locations and
            (in_idx in [3, 6, 7] and out_idx in [1, 5, 8])  # Wall exists left
        ):
            self._out_boundary[-1, :, out_idx] = True
            self._in_boundary[-1, :, in_idx] = True
        if (
            DirectionIndicators.TOP in self.boundary_locations and
            out_idx in [2, 5, 6] and in_idx in [4, 7, 8]  # Wall exists top
        ):
            self._out_boundary[:, -1, out_idx] = True
            self._in_boundary[:, -1, in_idx] = True
        if (
            DirectionIndicators.BOTTOM in self.boundary_locations and
            in_idx in [2, 5, 6] and out_idx in [4, 7, 8]  # Wall exists bottom
        ):
            self._out_boundary[:, 0, out_idx] = True
            self._in_boundary[:, 0, in_idx] = True

    def _init_boundary(self, pressure_variation: bool) -> None:
        assert not self._finish_initialize
        self._init_boundary_indices(pressure_variation)
        X, Y = self.out_boundary.shape[:-1]

        init_boundary = np.zeros((X, Y), np.bool8)
        """ init_boundary for the pressure_variation case """
        if DirectionIndicators.LEFT in self.boundary_locations:
            init_boundary[0, :] = np.ones(Y)
        if DirectionIndicators.RIGHT in self.boundary_locations:
            init_boundary[-1, :] = np.ones(Y)
        if DirectionIndicators.TOP in self.boundary_locations:
            init_boundary[:, -1] = np.ones(X)
        if DirectionIndicators.BOTTOM in self.boundary_locations:
            init_boundary[:, 0] = np.ones(X)

        for out_idx, in_idx in zip(self.out_indices, self.in_indices):
            if pressure_variation:
                self._out_boundary[:, :, out_idx] = init_boundary
                self._in_boundary[:, :, in_idx] = init_boundary
            else:
                self._allocate_boundary_conditions(in_idx=in_idx, out_idx=out_idx)

        if self._visualize_wall:
            self.visualize_wall_in_cui()

        self._finish_initialize = True

    def visualize_wall_in_cui(self, compress: bool = True) -> None:
        """ Wall visualizer for debugging """
        X, Y = self.out_boundary.shape[:-1]
        assert X >= 5 and Y >= 5
        y_itr = [Y - 1, Y - 2, Y // 2, 1, 0] if compress else range(Y - 1, -1, -1)
        x_itr = [0, 1, X // 2, X - 2, X - 1] if compress else range(X)

        child_cls = set([obj.__name__ for obj in self.__class__.__mro__])
        child_cls -= set(['BaseBoundary', 'AbstractBoundaryHandling', 'object'])
        boundary_name = list(child_cls)[0]

        print(f"### {boundary_name} Out boundary ###")
        for y in y_itr:
            display = ""
            for x in x_itr:
                display += get_direction_representor(boundary=self.out_boundary[x][y])
            print(display)

        print(f"\n### {boundary_name} In boundary ###")
        for y in y_itr:
            display = ""
            for x in x_itr:
                display += get_direction_representor(boundary=self.in_boundary[x][y])
            print(display)

        print("")


class RigidWall(BaseBoundary, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, boundary_locations: List[DirectionIndicators]):
        super().__init__(field, boundary_locations)

    def boundary_handling(self, field: FluidField2D) -> None:
        pdf_post = field.pdf
        pdf_post[self.in_boundary] = field.pdf_pre[self.out_boundary]


class MovingWall(BaseBoundary, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, boundary_locations: List[DirectionIndicators], wall_vel: np.ndarray):
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
        super().__init__(field, boundary_locations)
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

        pdf_post = field.pdf
        average_density = field.density.mean()

        pdf_post[self.in_boundary] = (
            field.pdf_pre[self.out_boundary]
            - average_density *
            self.weighted_vel_dot_wall_vel6[self.out_boundary]
        )


class PeriodicBoundaryConditions(BaseBoundary, AbstractBoundaryHandling):
    def __init__(self, field: FluidField2D, boundary_locations: List[DirectionIndicators],
                 in_density_factor: float, out_density_factor: float):

        super().__init__(field, boundary_locations, pressure_variation=True)

        # left to right (the flow of particles)
        self.horiz = (DirectionIndicators.LEFT in self.boundary_locations and
                      DirectionIndicators.RIGHT in self.boundary_locations)
        # bottom to top
        self.vert = (DirectionIndicators.TOP in self.boundary_locations and
                     DirectionIndicators.BOTTOM in self.boundary_locations)

        assert self.vert or self.horiz
        X, Y = field.lattice_grid_shape
        boundary_shape = Y if self.horiz else X
        self._in_density = 3 * in_density_factor * np.ones(boundary_shape)
        self._out_density = 3 * out_density_factor * np.ones(boundary_shape)

    @property
    def in_density(self) -> np.ndarray:
        return self._in_density

    @in_density.setter
    def in_density(self) -> None:
        raise NotImplementedError("in_density is not supposed to change from outside.")

    @property
    def out_density(self) -> np.ndarray:
        return self._out_density

    @out_density.setter
    def out_density(self) -> None:
        raise NotImplementedError("out_density is not supposed to change from outside.")

    def boundary_handling(self, field: FluidField2D) -> None:
        pdf_eq, pdf_post = field.pdf_eq, field.pdf

        if self.horiz:
            pdf_eq_in = local_equilibrium(velocity=field.velocity[-2], density=self.in_density).squeeze()
            pdf_post[0][:, self.out_indices] = pdf_eq_in[:, self.out_indices] + (
                pdf_post[-2][:, self.out_indices] - pdf_eq[-2][:, self.out_indices]
            )

            pdf_eq_out = local_equilibrium(velocity=field.velocity[1], density=self.out_density).squeeze()
            pdf_post[-1][:, self.in_indices] = pdf_eq_out[:, self.in_indices] + (
                pdf_post[1][:, self.in_indices] - pdf_eq[1][:, self.in_indices]
            )
        else:
            pdf_eq_in = local_equilibrium(velocity=field.velocity[:, -2], density=self.in_density).squeeze()
            pdf_post[:, 0, self.out_indices] = pdf_eq_in[:, self.out_indices] + (
                pdf_post[:, -2, self.out_indices] - pdf_eq[:, -2, self.out_indices]
            )

            pdf_eq_out = local_equilibrium(velocity=field.velocity[:, 1], density=self.out_density).squeeze()
            pdf_post[:, -1, self.in_indices] = pdf_eq_out[:, self.in_indices] + (
                pdf_post[:, 1, self.in_indices] - pdf_eq[:, 1, self.in_indices]
            )
