from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

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
                 pressure_variation: bool = False, **kwargs: Dict[str, Any]):

        self._out_indices = np.arange(9)
        self._in_indices = AdjacentAttributes.reflected_direction
        self._finish_initialize = False
        self._lattice_grid_shape = field.lattice_grid_shape
        self._boundary_locations = boundary_locations

        self._init_boundary_indices(pressure_variation)

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
        self._finish_initialize = True


class RigidWall(BaseBoundary):
    def __init__(self, field: LatticeBoltzmannMethod, boundary_locations: List[DirectionIndicators]):
        super().__init__(field, boundary_locations)

    def boundary_handling(self, field: LatticeBoltzmannMethod) -> None:
        pdf_post = field.pdf

        if DirectionIndicators.TOP in self.boundary_locations:
            pdf_post[:, -1, self.in_indices] = field.pdf_pre[:, -1, self.out_indices]
        if DirectionIndicators.BOTTOM in self.boundary_locations:
            pdf_post[:, 0, self.in_indices] = field.pdf_pre[:, 0, self.out_indices]
        if DirectionIndicators.LEFT in self.boundary_locations:
            pdf_post[0, :, self.in_indices] = field.pdf_pre[0, :, self.out_indices]
        if DirectionIndicators.RIGHT in self.boundary_locations:
            pdf_post[-1, :, self.in_indices] = field.pdf_pre[-1, :, self.out_indices]


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

        self._weighted_vel_dot_wall_vel6 = np.zeros((*self._lattice_grid_shape, 9), np.float32)
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

        pdf_post, pdf_pre = field.pdf, field.pdf_pre
        coef = self.wall_density * self.weighted_vel_dot_wall_vel6
        # self._compute_wall_density(field.pdf_pre, field.pdf, field.velocity)

        if DirectionIndicators.TOP in self.boundary_locations:
            pdf_post[:, -1, self.in_indices] = pdf_pre[:, -1, self.out_indices] - coef[:, -1, self.out_indices]
        if DirectionIndicators.BOTTOM in self.boundary_locations:
            pdf_post[:, 0, self.in_indices] = pdf_pre[:, 0, self.out_indices] - coef[:, 0, self.out_indices]
        if DirectionIndicators.LEFT in self.boundary_locations:
            pdf_post[0, :, self.in_indices] = pdf_pre[0, :, self.out_indices] - coef[0, :, self.out_indices]
        if DirectionIndicators.RIGHT in self.boundary_locations:
            pdf_post[-1, :, self.in_indices] = pdf_pre[-1, :, self.out_indices] - coef[-1, :, self.out_indices]


class PeriodicBoundaryConditionsWithPressureVariation(BaseBoundary):
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
        self._in_density = np.full(boundary_shape, 3 * in_density_factor)
        self._out_density = np.full(boundary_shape, 3 * out_density_factor)

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
        pdf_eq, pdf_pre = field.pdf_eq, field.pdf_pre

        if self.horiz:
            pdf_eq_in = local_equilibrium(velocity=field.velocity[-2], density=self.in_density).squeeze()
            pdf_pre[0, :, self.out_indices] = pdf_eq_in[:, self.out_indices].T + (
                pdf_pre[-2, :, self.out_indices] - pdf_eq[-2, :, self.out_indices]
            )

            pdf_eq_out = local_equilibrium(velocity=field.velocity[1], density=self.out_density).squeeze()
            pdf_pre[-1, :, self.in_indices] = pdf_eq_out[:, self.in_indices].T + (
                pdf_pre[1, :, self.in_indices] - pdf_eq[1, :, self.in_indices]
            )
        else:
            pdf_eq_in = local_equilibrium(velocity=field.velocity[:, -2], density=self.in_density).squeeze()
            pdf_pre[:, 0, self.out_indices] = pdf_eq_in[:, self.out_indices] + (
                pdf_pre[:, -2, self.out_indices] - pdf_eq[:, -2, self.out_indices]
            )

            pdf_eq_out = local_equilibrium(velocity=field.velocity[:, 1], density=self.out_density).squeeze()
            pdf_pre[:, -1, self.in_indices] = pdf_eq_out[:, self.in_indices] + (
                pdf_pre[:, 1, self.in_indices] - pdf_eq[:, 1, self.in_indices]
            )


class SequentialBoundaryHandlings:
    def __init__(self, *boundary_handlings: Optional[BaseBoundary]):
        PressurePBC = PeriodicBoundaryConditionsWithPressureVariation

        self.pressure_pbcs = []
        self.bounce_backs = []

        if boundary_handlings is not None:

            for boundary_handling in boundary_handlings:
                if isinstance(boundary_handling, PressurePBC):
                    self.pressure_pbcs.append(boundary_handling)
                elif boundary_handling is not None:
                    self.bounce_backs.append(boundary_handling)

    def __call__(self, field: LatticeBoltzmannMethod) -> None:
        for boundary_handling in self.pressure_pbcs:
            boundary_handling(field)

        field._pdf = deepcopy(field.pdf_pre)
        field.update_pdf()

        for boundary_handling in self.bounce_backs:  # type: ignore
            boundary_handling(field)
