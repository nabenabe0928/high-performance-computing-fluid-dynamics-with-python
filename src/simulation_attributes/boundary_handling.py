from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.simulation_attributes.lattice_boltzmann_method import (
    LatticeBoltzmannMethod,
    local_equilibrium
)
from src.utils.constants import AdjacentAttributes, DirectionIndicators


def get_direction_representor(boundary: np.ndarray) -> str:
    indices = [i for i, b in enumerate(boundary) if b]
    """
    Adjacent cell indices
    6 2 5
    3 0 1
    7 4 8
    """
    if indices == list(AdjacentAttributes.x_left):
        return "<"
    if indices == list(AdjacentAttributes.x_right):
        return ">"
    if indices == list(AdjacentAttributes.y_top):
        return "^"
    if indices == list(AdjacentAttributes.y_bottom):
        return "v"
    else:
        return "*"


class BaseBoundary():
    def __init__(self, field: LatticeBoltzmannMethod, boundary_locations: List[DirectionIndicators],
                 pressure_variation: bool = False, visualize_wall: bool = False,
                 **kwargs: Dict[str, Any]):

        self._out_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._out_indices = np.arange(9)
        self._in_boundary = np.zeros((*field.lattice_grid_shape, 9), np.bool8)
        self._in_indices = AdjacentAttributes.reflected_direction
        self._finish_initialize = False
        self._lattice_grid_shape = field.lattice_grid_shape
        self._boundary_locations = boundary_locations
        self._visualize_wall = visualize_wall

        self._init_boundary(pressure_variation=pressure_variation)

    def __call__(self, field: LatticeBoltzmannMethod) -> None:
        self.boundary_handling(field)

    @abstractmethod
    def boundary_handling(self, field: LatticeBoltzmannMethod) -> None:
        """
        Compute the PDF using pdf_pre, pdf_mid, pdf and density, velocity
        and return the PDF after boundary handling.
        In order to be able to joint multiple boundary handlings,
        the update of the PDF has to be performed inside the function.

        Args:
            field (LatticeBoltzmannMethod)
        """
        raise NotImplementedError("The child class of BaseBoundary must have boundary_handling function.")

    @property
    def in_boundary(self) -> np.ndarray:
        """
        Returns:
            _in_boundary (np.ndarray):
                Direction to come in.
                In other words, if the reflected direction of _out_boundary
                or the inlet of pipes.
                Each element is True or False with the shape of (X, Y, 9).
        """
        return self._in_boundary

    @in_boundary.setter
    def in_boundary(self) -> None:
        raise NotImplementedError("in_boundary is not supposed to change from outside.")

    @property
    def out_boundary(self) -> np.ndarray:
        """
        Returns:
            _out_boundary (np.ndarray):
                Direction to come out.
                In other words, if there are walls (or boundary) or not
                or the outlet of pipes.
                Each element is True or False with the shape of (X, Y, 9).
        """
        return self._out_boundary

    @out_boundary.setter
    def out_boundary(self) -> None:
        raise NotImplementedError("out_boundary is not supposed to change from outside.")

    @property
    def in_indices(self) -> np.ndarray:
        """
        Returns:
            _in_indices (np.ndarray):
                The corresponding indices for the bouncing direction of _out_indices.
                The shape is (n_direction, ).
        """
        return self._in_indices

    @in_indices.setter
    def in_indices(self) -> None:
        raise NotImplementedError("in_indices is not supposed to change from outside.")

    @property
    def out_indices(self) -> np.ndarray:
        """
        Returns:
            _out_indices (np.ndarray):
                It stands for which directions (from 9 adjacent cells) can have the out-boundary.
                The shape is (n_direction, ) where n_direction is smaller than 9.
        """
        return self._out_indices

    @out_indices.setter
    def out_indices(self) -> None:
        raise NotImplementedError("out_indices is not supposed to change from outside.")

    @property
    def boundary_locations(self) -> List[DirectionIndicators]:
        """
        Returns:
            _boundary_locations (List[DirectionIndicators]):
                Which direction we have the walls.
        """
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
                out_indices += list(AdjacentAttributes.x_left)
            if DirectionIndicators.RIGHT in self.boundary_locations:
                out_indices += list(AdjacentAttributes.x_right)
            if DirectionIndicators.BOTTOM in self.boundary_locations:
                out_indices += list(AdjacentAttributes.y_bottom)
            if DirectionIndicators.TOP in self.boundary_locations:
                out_indices += list(AdjacentAttributes.y_top)
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
                out_indices = list(AdjacentAttributes.x_right)
            else:
                # bottom to top
                out_indices = list(AdjacentAttributes.y_top)

        self._out_indices = np.array(out_indices)
        self._in_indices = self.in_indices[self.out_indices]

    def _allocate_boundary_conditions(self, in_idx: int, out_idx: int) -> None:
        """
        Based on the indices of the adjacent cell indices:
        6 2 5
        3 0 1
        7 4 8,
        we give True or False to each grid (x, y) if (x, y) has the boundary.

        Args:
            in_idx (int):
                The direction to come in from the outside.
            out_idx (int):
                The direction to come out from the inside.
        """
        left = list(AdjacentAttributes.x_left)
        right = list(AdjacentAttributes.x_right)
        top = list(AdjacentAttributes.y_top)
        bottom = list(AdjacentAttributes.y_bottom)

        if (
            DirectionIndicators.LEFT in self.boundary_locations and
            (out_idx in left and in_idx in right)  # Wall exists left
        ):
            self._out_boundary[0, :, out_idx] = True
            self._in_boundary[0, :, in_idx] = True
        if (
            DirectionIndicators.RIGHT in self.boundary_locations and
            (in_idx in left and out_idx in right)  # Wall exists left
        ):
            self._out_boundary[-1, :, out_idx] = True
            self._in_boundary[-1, :, in_idx] = True
        if (
            DirectionIndicators.TOP in self.boundary_locations and
            out_idx in top and in_idx in bottom  # Wall exists top
        ):
            self._out_boundary[:, -1, out_idx] = True
            self._in_boundary[:, -1, in_idx] = True
        if (
            DirectionIndicators.BOTTOM in self.boundary_locations and
            in_idx in top and out_idx in bottom  # Wall exists bottom
        ):
            self._out_boundary[:, 0, out_idx] = True
            self._in_boundary[:, 0, in_idx] = True

    def _init_boundary(self, pressure_variation: bool) -> None:
        """
        Initialize the boundary of the shape (X, Y)
        based on the feeded boundary locations.

        Args:
            pressure_variation (bool):
                if the simulation is based on pressure variation.
                If True, the initialization is slightly different.
        """
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
        child_cls -= set(['BaseBoundary', 'object'])
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


def sequential_boundary_handlings(*boundary_handlings: Optional[BaseBoundary]
                                  ) -> Callable[[LatticeBoltzmannMethod], None]:

    def _imp(field: LatticeBoltzmannMethod) -> None:
        for boundary_handling in boundary_handlings:
            if boundary_handling is not None:
                boundary_handling(field)

    return _imp


class RigidWall(BaseBoundary):
    def __init__(self, field: LatticeBoltzmannMethod, boundary_locations: List[DirectionIndicators]):
        super().__init__(field, boundary_locations)

    def boundary_handling(self, field: LatticeBoltzmannMethod) -> None:
        pdf_post = field.pdf
        pdf_post[self.in_boundary] = field.pdf_pre[self.out_boundary]


def dir2coef(wall: DirectionIndicators, dir: DirectionIndicators, equilibrium: bool = False) -> float:
    if dir.is_opposite(wall):
        return 0.0
    elif dir.is_sameside(wall):
        return 1.0
    else:
        return not equilibrium


class MovingWall(BaseBoundary):
    coefs_pre = {
        wall: [dir2coef(wall, dir, True) for dir in DirectionIndicators]
        for wall in DirectionIndicators
    }
    coefs_post = {
        wall: [dir2coef(wall, dir, False) for dir in DirectionIndicators]
        for wall in DirectionIndicators
    }

    def __init__(self, field: LatticeBoltzmannMethod,
                 boundary_locations: List[DirectionIndicators], wall_vel: np.ndarray):
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
        self._wall_density = np.full((*field.lattice_grid_shape, 9), field.density_avg)

        if len(boundary_locations) != 1:
            raise ValueError("Moving wall only supports one moving wall, but got {} directions".format(
                len(boundary_locations)
            ))

        self._prod_coef_pre = np.array(self.coefs_pre[boundary_locations[0]])
        self._prod_coef_post = np.array(self.coefs_post[boundary_locations[0]])

    @property
    def wall_vel(self) -> np.ndarray:
        """ The velocity of the wall """
        return self._wall_vel

    @wall_vel.setter
    def wall_vel(self) -> None:
        """ The velocity of the wall """
        raise NotImplementedError("wall_vel is not supposed to change from outside.")

    @property
    def wall_density(self) -> np.ndarray:
        """ The density at the wall """
        return self._wall_density

    @wall_density.setter
    def wall_density(self) -> None:
        raise NotImplementedError("wall_density is not supposed to change from outside.")

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

    def _compute_wall_density(self, pdf_pre: np.ndarray, pdf: np.ndarray, vel: np.ndarray) -> None:
        """
        The computation of the average density at the wall follows the following literatures:
            Title: A study of wall boundary conditions in pseudopotential lattice Boltzmann models
            Authors: Sorush Khajepor et al.
            Equation number: 14

            Title: On pressure and velocity boundary conditions for the lattice Boltzmann BGK model
            Authors: Qisu Zou and Xiaoyi He
            Equation number: 19
        """
        wall = self.boundary_locations[0]
        if wall == DirectionIndicators.RIGHT:
            wall_density = pdf_pre[-1, ...] @ self._prod_coef_pre
            wall_density += pdf[-1, ...] @ self._prod_coef_post
            self._wall_density[-1, ...] = wall_density[:, np.newaxis]
        elif wall == DirectionIndicators.LEFT:
            wall_density = pdf_pre[0, ...] @ self._prod_coef_pre
            wall_density += pdf[0, ...] @ self._prod_coef_post
            self._wall_density[0, ...] = wall_density[:, np.newaxis]
        elif wall == DirectionIndicators.TOP:
            wall_density = pdf_pre[:, -1, ...] @ self._prod_coef_pre
            wall_density += pdf[:, -1, ...] @ self._prod_coef_post
            self._wall_density[:, -1, ...] = wall_density[:, np.newaxis]
        elif wall == DirectionIndicators.BOTTOM:
            wall_density = pdf_pre[:, 0, ...] @ self._prod_coef_pre
            wall_density += pdf[:, 0, ...] @ self._prod_coef_post
            self._wall_density[:, 0, ...] = wall_density[:, np.newaxis]

    def boundary_handling(self, field: LatticeBoltzmannMethod) -> None:
        """
        There are two ways to compute the wall density:
            1. Extrapolation as in _compute_wall_density()
            2. Average density
                In this case, the total mass is preverved,
                so always the initial average density.

        We use 2. in this implementation for more stability.
        """
        if not self._finish_precompute:
            self._precompute()

        pdf_post = field.pdf
        # self._compute_wall_density(field.pdf_pre, field.pdf, field.velocity)

        pdf_post[self.in_boundary] = (
            field.pdf_pre[self.out_boundary]
            - self.wall_density[self.out_boundary] *
            self.weighted_vel_dot_wall_vel6[self.out_boundary]
        )


class PeriodicBoundaryConditions(BaseBoundary):
    def __init__(self, field: LatticeBoltzmannMethod, boundary_locations: List[DirectionIndicators],
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

    def boundary_handling(self, field: LatticeBoltzmannMethod) -> None:
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
