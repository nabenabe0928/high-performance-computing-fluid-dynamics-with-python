import numpy as np
from tqdm import trange
from typing import Tuple

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import PeriodicBoundaryConditions, RigidWall
from src.utils.attr_dict import AttrDict
from src.utils.constants import DirectionIndicators
from src.utils.visualization import visualize_velocity_field_of_pipe


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    in_density_factor: float = (1. + 3e-3) / 3.
    out_density_factor: float = 1. / 3.
    lattice_grid_shape: Tuple[int, int] = (30, 30)


def main(init_density: np.ndarray, init_velocity: np.ndarray,
         total_time_steps: int, omega: float, lattice_grid_shape: Tuple[int, int],
         in_density_factor: float, out_density_factor: float) -> None:

    X, Y = lattice_grid_shape

    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity, init_density=init_density)

    pbc = PeriodicBoundaryConditions(
        field=field,
        boundary_locations=[DirectionIndicators.LEFT, DirectionIndicators.RIGHT],
        in_density_factor=in_density_factor,
        out_density_factor=out_density_factor
    )

    rigid_wall = RigidWall(
        field=field,
        boundary_locations=[DirectionIndicators.TOP, DirectionIndicators.BOTTOM]
    )

    def boundary_handling_func(field: LatticeBoltzmannMethod) -> None:
        pbc.boundary_handling(field)
        rigid_wall.boundary_handling(field)

    field.local_equilibrium_pdf_update()
    for t in trange(total_time_steps):
        field.lattice_boltzmann_step(boundary_handling=boundary_handling_func)

    visualize_velocity_field_of_pipe(field=field, pbc=pbc)


if __name__ == '__main__':
    kwargs = ExperimentVariables(
        omega=1.5,
        total_time_steps=5000,
        lattice_grid_shape=(30, 30)
    )
    X, Y = kwargs.lattice_grid_shape

    density, vel = np.ones((X, Y)), np.zeros((X, Y, 2))
    main(init_density=density, init_velocity=vel, **kwargs)
