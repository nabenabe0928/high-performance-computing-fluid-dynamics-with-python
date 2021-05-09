import numpy as np

from src.milestones.constants import Milestone2InitVals
# from src.utils.visualization import visualize_density_surface
from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod


lattice_grid_shape = (50, 50)
init_vals = Milestone2InitVals()


def main(total_time_steps: int, init_density: np.ndarray, init_vel: np.ndarray) -> None:
    X, Y = lattice_grid_shape
    field = LatticeBoltzmannMethod(X, Y, init_density=init_density, init_vel=init_vel)
    field(total_time_steps)
    # visualize_density_surface(field)


if __name__ == '__main__':
    init_density, init_vel = init_vals.test1(lattice_grid_shape)
    main(total_time_steps=70, init_density=init_density, init_vel=init_vel)

    init_density, init_vel = init_vals.test2(lattice_grid_shape)
    main(total_time_steps=10000, init_density=init_density, init_vel=init_vel)
