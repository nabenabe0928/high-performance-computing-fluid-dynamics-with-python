import numpy as np
from tqdm import trange

from src.milestones.constants import Milestone2InitVals
from src.utils.visualization import visualize_density_surface
from src.simulation_attributes.formula import FluidField2D


lattice_grid_shape = (50, 50)
init_vals = Milestone2InitVals()


def main(total_time_steps: int, init_density: np.ndarray, init_vel: np.ndarray) -> None:
    X, Y = lattice_grid_shape
    field = FluidField2D(X, Y)
    field.init_vals(init_density=init_density, init_vel=init_vel)

    field.local_equilibrium_pdf_update()
    for _ in trange(total_time_steps):
        field.lattice_boltzmann_step()

    visualize_density_surface(field)


if __name__ == '__main__':
    init_density, init_vel = init_vals.test1(lattice_grid_shape)
    main(total_time_steps=70, init_density=init_density, init_vel=init_vel)

    init_density, init_vel = init_vals.test2(lattice_grid_shape)
    main(total_time_steps=10000, init_density=init_density, init_vel=init_vel)
