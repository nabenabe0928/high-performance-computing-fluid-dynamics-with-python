import numpy as np
from tqdm import trange

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import MovingWall, RigidWall
from src.utils.attr_dict import AttrDict
from src.utils.constants import DirectionIndicators
from src.utils.parallel_computation import ChunkedGridManager
from src.utils.visualization import visualize_velocity_field_mpi


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    wall_vel: np.ndarray = np.array([10, 0])


lattice_grid_shape = (90, 90)


def main(init_density: np.ndarray, init_velocity: np.ndarray, grid_manager: ChunkedGridManager,
         total_time_steps: int, omega: float, wall_vel: np.ndarray) -> None:
    X, Y = grid_manager.local_grid_size

    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity,
                                   init_density=init_density, grid_manager=grid_manager)

    boundary_locations = [
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
    if len(boundary_locations) >= 1:
        rigid_wall = RigidWall(
            field,
            boundary_locations=boundary_locations
        )

    def boundary_handling_func(field: LatticeBoltzmannMethod) -> None:
        if rigid_wall is not None:
            rigid_wall.boundary_handling(field)
        if moving_wall is not None:
            moving_wall.boundary_handling(field)

    field.local_equilibrium_pdf_update()
    for t in trange(total_time_steps):
        field.lattice_boltzmann_step(boundary_handling=boundary_handling_func)

    x_file, y_file = 'ux.npy', 'uy.npy'
    field.grid_manager.save_mpiio(x_file, field.velocity[..., 0])
    field.grid_manager.save_mpiio(y_file, field.velocity[..., 1])

    if field.grid_manager.rank == 0:
        visualize_velocity_field_mpi(x_file, y_file)


if __name__ == '__main__':
    # Reynolds = 300 * wall_vel / viscosity
    viscosity = 1. / 3.
    kwargs = ExperimentVariables(
        omega=1. / (3. * viscosity + 0.5),
        total_time_steps=100,
        wall_vel=np.array([.1, 0])
    )
    grid_manager = ChunkedGridManager(*lattice_grid_shape)

    buffer_grid_shape = grid_manager.buffer_grid_size
    local_grid_size = grid_manager.local_grid_size
    print(f"{grid_manager.rank} / {grid_manager.size}: {local_grid_size}")

    density, vel = np.ones(buffer_grid_shape), np.zeros((*buffer_grid_shape, 2))
    main(init_density=density, init_velocity=vel, grid_manager=grid_manager, **kwargs)
