import numpy as np
from tqdm import trange

from src.simulation_attributes.formula import FluidField2D
from src.simulation_attributes.boundary_handling import PeriodicBoundaryConditions, RigidWall
from src.utils.attr_dict import AttrDict
from src.utils.visualization import visualize_velocity_field_of_pipe


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    in_density_factor: float = (1. + 3e-3) / 3.
    out_density_factor: float = 1. / 3.


lattice_grid_shape = (30, 30)


def main(init_density: np.ndarray, init_velocity: np.ndarray,
         total_time_steps: int, omega: float,
         in_density_factor: float, out_density_factor: float) -> None:

    X, Y = lattice_grid_shape

    field = FluidField2D(X, Y, omega=omega, init_vel=init_velocity, init_density=init_density)

    init_pbc_boundary = np.zeros((X, Y))
    init_pbc_boundary[0, :] = np.ones(Y)
    init_pbc_boundary[-1, :] = np.ones(Y)

    pbc = PeriodicBoundaryConditions(field=field, init_boundary=init_pbc_boundary,
                                     in_density_factor=in_density_factor,
                                     out_density_factor=out_density_factor)

    init_rigid_boundary = np.zeros((X, Y))
    init_rigid_boundary[:, 0] = np.ones(X)
    init_rigid_boundary[:, -1] = np.ones(X)
    rigid_wall = RigidWall(field, init_boundary=init_rigid_boundary)

    def boundary_handling_func(field: FluidField2D) -> None:
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
    )

    density, vel = np.ones(lattice_grid_shape), np.zeros((*lattice_grid_shape, 2))
    main(init_density=density, init_velocity=vel, **kwargs)
