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
from src.utils.parallel_computation import ChunkedGridManager
from src.utils.visualization import visualize_velocity_field_mpi


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    wall_vel: np.ndarray = np.array([10, 0])
    lattice_grid_shape: Tuple[int, int] = (100, 100)


def main(init_density: np.ndarray, init_velocity: np.ndarray, grid_manager: ChunkedGridManager,
         lattice_grid_shape: Tuple[int, int], total_time_steps: int, omega: float, wall_vel: np.ndarray
         ) -> None:
    X, Y = grid_manager.local_grid_size

    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity,
                                   init_density=init_density, grid_manager=grid_manager)

    rigid_boundary_locations = [
        getattr(DirectionIndicators, dir)
        for dir in ['TOP', 'LEFT', 'RIGHT']
        if grid_manager.is_boundary(getattr(DirectionIndicators, dir))
    ]

    moving_wall, rigid_wall = None, None
    if grid_manager.is_boundary(DirectionIndicators.BOTTOM):
        moving_wall = MovingWall(
            field,
            boundary_locations=[DirectionIndicators.BOTTOM],
            wall_vel=wall_vel
        )
    if len(rigid_boundary_locations) >= 1:
        rigid_wall = RigidWall(
            field,
            boundary_locations=rigid_boundary_locations
        )

    field(total_time_steps, boundary_handling=sequential_boundary_handlings(rigid_wall, moving_wall))

    x_file, y_file = field.save_velocity_field(
        vis_name='test_run',
        file_name='v',
        index=total_time_steps
    )

    if field.grid_manager.rank == 0:
        visualize_velocity_field_mpi(x_file, y_file)


if __name__ == '__main__':
    # Reynolds = 300 * wall_vel / viscosity
    # 0.03 < viscosity < inf
    viscosity = 1. / 30.
    kwargs = ExperimentVariables(
        omega=1. / (3. * viscosity + 0.5),
        total_time_steps=100,
        wall_vel=np.array([.3, 0]),
        lattice_grid_shape=(100, 100)
    )
    grid_manager = ChunkedGridManager(*kwargs.lattice_grid_shape)

    buffer_grid_size = grid_manager.buffer_grid_size

    density, vel = np.ones(buffer_grid_size), np.zeros((*buffer_grid_size, 2))
    main(init_density=density, init_velocity=vel, grid_manager=grid_manager, **kwargs)
