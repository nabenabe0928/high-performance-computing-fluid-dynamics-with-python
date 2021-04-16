import numpy as np
from tqdm import trange

from src.milestones.constants import sinusoidal_density, sinusoidal_velocity
from src.utils.visualization import visualize_density_surface
from src.utils.formula import FluidField2D


lattice_grid_shape = (50, 50)
epsilon, rho0, omega = 0.01, 0.5, 0.5


def main(init_density: np.ndarray, init_velocity: np.ndarray, total_time_steps: int) -> None:
    X, Y = lattice_grid_shape
    field = FluidField2D(X, Y, omega=omega)
    field.init_vals(init_vel=init_velocity, init_density=init_density)

    for _ in trange(total_time_steps):
        field.lattice_boltzmann_step()


if __name__ == '__main__':
    density, vel = sinusoidal_density(lattice_grid_shape, epsilon=0.01, rho0=0.5)
    main(init_density=density, init_velocity=vel)
    density, vel = sinusoidal_velocity(lattice_grid_shape, epsilon=0.01)
    main(init_density=density, init_velocity=vel)
