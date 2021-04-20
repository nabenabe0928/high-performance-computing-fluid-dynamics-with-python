import numpy as np
from tqdm import trange

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import DirectionIndicators, MovingWall, RigidWall
from src.utils.attr_dict import AttrDict
from src.utils.visualization import visualize_velocity_field_of_moving_wall


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    wall_vel: np.ndarray = np.array([10, 0])


lattice_grid_shape = (50, 50)


def main(init_density: np.ndarray, init_velocity: np.ndarray,
         total_time_steps: int, omega: float, wall_vel: np.ndarray) -> None:
    X, Y = lattice_grid_shape

    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity, init_density=init_density)

    rigid_wall = RigidWall(
        field=field,
        boundary_locations=[DirectionIndicators.TOP]
    )
    moving_wall = MovingWall(
        field=field,
        boundary_locations=[DirectionIndicators.BOTTOM],
        wall_vel=wall_vel
    )

    def boundary_handling_func(field: LatticeBoltzmannMethod) -> None:
        rigid_wall.boundary_handling(field)
        moving_wall.boundary_handling(field)

    field.local_equilibrium_pdf_update()
    for t in trange(total_time_steps):
        field.lattice_boltzmann_step(boundary_handling=boundary_handling_func)

    visualize_velocity_field_of_moving_wall(field=field, wall_vel=wall_vel)


if __name__ == '__main__':
    kwargs = ExperimentVariables(
        omega=0.5,
        total_time_steps=5000,
        wall_vel=np.array([10, 0])
    )

    density, vel = np.ones(lattice_grid_shape), np.zeros((*lattice_grid_shape, 2))
    main(init_density=density, init_velocity=vel, **kwargs)
