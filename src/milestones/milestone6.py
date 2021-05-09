import numpy as np
from typing import Tuple

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import (
    MovingWall,
    RigidWall,
    sequential_boundary_handlings
)
from src.utils.utils import AttrDict
from src.utils.constants import DirectionIndicators
# from src.utils.visualization import visualize_velocity_field


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    wall_vel: np.ndarray = np.array([10, 0])
    lattice_grid_shape: Tuple[int, int] = (100, 100)


def main(init_density: np.ndarray, init_velocity: np.ndarray, lattice_grid_shape: Tuple[int, int],
         total_time_steps: int, omega: float, wall_vel: np.ndarray) -> None:
    X, Y = lattice_grid_shape

    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity, init_density=init_density)

    moving_wall = MovingWall(
        field,
        boundary_locations=[DirectionIndicators.BOTTOM],
        wall_vel=wall_vel
    )

    rigid_wall = RigidWall(
        field,
        boundary_locations=[
            DirectionIndicators.TOP,
            DirectionIndicators.LEFT,
            DirectionIndicators.RIGHT
        ]
    )

    field(total_time_steps, boundary_handling=sequential_boundary_handlings(rigid_wall, moving_wall))
    # visualize_velocity_field(field=field)


if __name__ == '__main__':
    # Reynolds = 300 * wall_vel / viscosity
    viscosity = 1. / 30.
    kwargs = ExperimentVariables(
        omega=1. / (3. * viscosity + 0.5),
        total_time_steps=100,
        wall_vel=np.array([.3, 0]),
        lattice_grid_shape=(100, 100)
    )

    density, vel = np.ones(kwargs.lattice_grid_shape), np.zeros((*kwargs.lattice_grid_shape, 2))
    main(init_density=density, init_velocity=vel, **kwargs)
