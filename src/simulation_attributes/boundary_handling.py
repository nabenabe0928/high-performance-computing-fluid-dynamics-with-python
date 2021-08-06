from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from src.simulation_attributes.lattice_boltzmann_method import (
    LatticeBoltzmannMethod,
    local_equilibrium
)
from src.utils.constants import AdjacentAttributes, DirectionIndicators


class BaseBoundary():
    def __init__(self, field: LatticeBoltzmannMethod, boundary_locations: List[DirectionIndicators],
                 pressure_variation: bool = False, **kwargs: Dict[str, Any]):

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
    def in_indices(self) -> Dict[DirectionIndicators, np.ndarray]:
        """
        Returns:
            _in_indices (np.ndarray):
                The corresponding indices for the bouncing direction of _out_indices.
        """
        return self._in_indices

    @in_indices.setter
    def in_indices(self) -> None:
        raise NotImplementedError("in_indices is not supposed to change from outside.")

    @property
    def out_indices(self) -> Dict[DirectionIndicators, np.ndarray]:
        """
        Returns:
            _out_indices (Dict[DirectionIndicators, np.ndarray]):
                It stands for which directions (from 9 adjacent cells) can have the out-boundary.
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
        rd = AdjacentAttributes.reflected_direction
        out_indices, in_indices = {}, {}
        if not pressure_variation:
            if DirectionIndicators.LEFT in self.boundary_locations:
                dirs = AdjacentAttributes.x_left
                out_indices[DirectionIndicators.LEFT] = dirs
                in_indices[DirectionIndicators.LEFT] = rd[dirs]
            if DirectionIndicators.RIGHT in self.boundary_locations:
                dirs = AdjacentAttributes.x_right
                out_indices[DirectionIndicators.RIGHT] = dirs
                in_indices[DirectionIndicators.RIGHT] = rd[dirs]
            if DirectionIndicators.BOTTOM in self.boundary_locations:
                dirs = AdjacentAttributes.y_bottom
                out_indices[DirectionIndicators.BOTTOM] = dirs
                in_indices[DirectionIndicators.BOTTOM] = rd[dirs]
            if DirectionIndicators.TOP in self.boundary_locations:
                dirs = AdjacentAttributes.y_top
                out_indices[DirectionIndicators.TOP] = dirs
                in_indices[DirectionIndicators.TOP] = rd[dirs]
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
                dirs = AdjacentAttributes.x_right
                out_indices[DirectionIndicators.RIGHT] = dirs
                in_indices[DirectionIndicators.RIGHT] = rd[dirs]
            else:
                # bottom to top
                dirs = AdjacentAttributes.y_top
                out_indices[DirectionIndicators.TOP] = dirs
                in_indices[DirectionIndicators.TOP] = rd[dirs]

        self._out_indices = out_indices
        self._in_indices = in_indices
        self._finish_initialize = True


class RigidWall(BaseBoundary):
    def __init__(self, field: LatticeBoltzmannMethod, boundary_locations: List[DirectionIndicators]):
        super().__init__(field, boundary_locations)

    def __repr__(self) -> str:
        repr = 'RigidWall('
        repr = '{}boundary_locations={}, in_indices={}, out_indices={})'.format(
            repr,
            self.boundary_locations,
            self.in_indices,
            self.out_indices
        )

        return repr

    def boundary_handling(self, field: LatticeBoltzmannMethod) -> None:
        pdf_post, pdf_pre = field.pdf, field.pdf_pre

        if DirectionIndicators.TOP in self.boundary_locations:
            dir = DirectionIndicators.TOP
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[:, -1, in_idx] = pdf_pre[:, -1, out_idx]
        if DirectionIndicators.BOTTOM in self.boundary_locations:
            dir = DirectionIndicators.BOTTOM
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[:, 0, in_idx] = pdf_pre[:, 0, out_idx]
        if DirectionIndicators.LEFT in self.boundary_locations:
            dir = DirectionIndicators.LEFT
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[0, :, in_idx] = pdf_pre[0, :, out_idx]
        if DirectionIndicators.RIGHT in self.boundary_locations:
            dir = DirectionIndicators.RIGHT
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[-1, :, in_idx] = pdf_pre[-1, :, out_idx]


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
                 boundary_locations: List[DirectionIndicators],
                 wall_vel: np.ndarray,
                 extrapolation: bool = False):
        """
        Attributes:
            _wall_vel (np.ndarray):
                The velocity vector of the movement of the wall

            _weighted_vel_dot_wall_vel6 (np.ndarray):
                The computation results of
                2 * wi * rhow * (ci @ uw) / cs ** 2
                in the equation for the moving wall.
            _extrapolation (bool):
                If using extrapolation for the wall density or not.

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
        self._extrapolation = extrapolation

        if len(boundary_locations) != 1:
            raise ValueError("Moving wall only supports one moving wall, but got {} directions".format(
                len(boundary_locations)
            ))

        self._prod_coef_pre = np.array(self.coefs_pre[boundary_locations[0]])
        self._prod_coef_post = np.array(self.coefs_post[boundary_locations[0]])

    def __repr__(self) -> str:
        repr = 'MovingWall('
        repr = '{}boundary_locations={}, in_indices={}, out_indices={}, wall_vel={}, extrapolation={})'.format(
            repr,
            self.boundary_locations,
            self.in_indices,
            self.out_indices,
            self.wall_vel,
            self.extrapolation
        )

        return repr

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
    def extrapolation(self) -> bool:
        return self._extrapolation

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
        ws = AdjacentAttributes.weights
        vs = AdjacentAttributes.velocity_direction_set

        self._weighted_vel_dot_wall_vel6 = np.zeros((*self._lattice_grid_shape, 9), np.float32)

        if DirectionIndicators.TOP in self.boundary_locations:
            out_idx = self.out_indices[DirectionIndicators.TOP]
            w, v = ws[out_idx], vs[out_idx]
            value = 6 * w * (v @ self.wall_vel)
            self._weighted_vel_dot_wall_vel6[:, -1, out_idx] = value[np.newaxis, :]
        if DirectionIndicators.BOTTOM in self.boundary_locations:
            out_idx = self.out_indices[DirectionIndicators.BOTTOM]
            w, v = ws[out_idx], vs[out_idx]
            value = 6 * w * (v @ self.wall_vel)
            self._weighted_vel_dot_wall_vel6[:, 0, out_idx] = value[np.newaxis, :]
        if DirectionIndicators.LEFT in self.boundary_locations:
            out_idx = self.out_indices[DirectionIndicators.LEFT]
            w, v = ws[out_idx], vs[out_idx]
            value = 6 * w * (v @ self.wall_vel)
            self._weighted_vel_dot_wall_vel6[0, :, out_idx] = value[np.newaxis, :].T
        if DirectionIndicators.RIGHT in self.boundary_locations:
            out_idx = self.out_indices[DirectionIndicators.RIGHT]
            w, v = ws[out_idx], vs[out_idx]
            value = 6 * w * (v @ self.wall_vel)
            self._weighted_vel_dot_wall_vel6[-1, :, out_idx] = value[np.newaxis, :].T

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

        We use 2. in this implementation for more stability by default.
        """
        if not self._finish_precompute:
            self._precompute()

        pdf_post, pdf_pre = field.pdf, field.pdf_pre

        if self.extrapolation:
            self._compute_wall_density(field.pdf_pre, field.pdf, field.velocity)

        coef = self.wall_density * self.weighted_vel_dot_wall_vel6

        if DirectionIndicators.TOP in self.boundary_locations:
            dir = DirectionIndicators.TOP
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[:, -1, in_idx] = pdf_pre[:, -1, out_idx] - coef[:, -1, out_idx]
        if DirectionIndicators.BOTTOM in self.boundary_locations:
            dir = DirectionIndicators.BOTTOM
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[:, 0, in_idx] = pdf_pre[:, 0, out_idx] - coef[:, 0, out_idx]
        if DirectionIndicators.LEFT in self.boundary_locations:
            dir = DirectionIndicators.LEFT
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[0, :, in_idx] = pdf_pre[0, :, out_idx] - coef[0, :, out_idx]
        if DirectionIndicators.RIGHT in self.boundary_locations:
            dir = DirectionIndicators.RIGHT
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]
            pdf_post[-1, :, in_idx] = pdf_pre[-1, :, out_idx] - coef[-1, :, out_idx]


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
        assert in_density_factor > out_density_factor
        X, Y = field.lattice_grid_shape
        boundary_shape = Y if self.horiz else X
        self._in_density = np.full(boundary_shape, 3 * in_density_factor)
        self._out_density = np.full(boundary_shape, 3 * out_density_factor)

    def __repr__(self) -> str:
        repr = 'PeriodicBoundaryConditionsWithPressureVariation('
        repr = '{}boundary_locations={}, in_indices={}, out_indices={}, in_density={}, out_density={})'.format(
            repr,
            self.boundary_locations,
            self.in_indices,
            self.out_indices,
            self.in_density,
            self.out_density
        )

        return repr

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
            dir = DirectionIndicators.RIGHT
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]

            pdf_eq_in = local_equilibrium(velocity=field.velocity[-2], density=self.in_density).squeeze()
            pdf_pre[0, :, out_idx] = pdf_eq_in[:, out_idx].T + (
                pdf_pre[-2, :, out_idx] - pdf_eq[-2, :, out_idx]
            )

            pdf_eq_out = local_equilibrium(velocity=field.velocity[1], density=self.out_density).squeeze()
            pdf_pre[-1, :, in_idx] = pdf_eq_out[:, in_idx].T + (
                pdf_pre[1, :, in_idx] - pdf_eq[1, :, in_idx]
            )
        else:
            dir = DirectionIndicators.TOP
            in_idx, out_idx = self.in_indices[dir], self.out_indices[dir]

            pdf_eq_in = local_equilibrium(velocity=field.velocity[:, -2], density=self.in_density).squeeze()
            pdf_pre[:, 0, out_idx] = pdf_eq_in[:, out_idx] + (
                pdf_pre[:, -2, out_idx] - pdf_eq[:, -2, out_idx]
            )

            pdf_eq_out = local_equilibrium(velocity=field.velocity[:, 1], density=self.out_density).squeeze()
            pdf_pre[:, -1, in_idx] = pdf_eq_out[:, in_idx] + (
                pdf_pre[:, 1, in_idx] - pdf_eq[:, 1, in_idx]
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

    def __repr__(self) -> str:
        repr = 'SequentialBoundaryHandlings(\n'
        cnt = 1
        for boundary_handling in self.pressure_pbcs:
            if boundary_handling is not None:
                repr += f'\t({cnt}): {str(boundary_handling)}\n'
                cnt += 1

        for boundary_handling in self.bounce_backs:  # type: ignore
            if boundary_handling is not None:
                repr += f'\t({cnt}): {str(boundary_handling)}\n'
                cnt += 1

        return f'{repr})'
