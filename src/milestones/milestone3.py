import numpy as np
from tqdm import trange
from typing import Callable, Tuple

from src.milestones.constants import sinusoidal_density, sinusoidal_velocity
from src.utils.visualization import visualize_density_surface, visualize_quantity_vs_time
from src.utils.formula import FluidField2D
from src.utils.attr_dict import AttrDict
from src.utils.constants import EquationFuncType


class ExperimentVariables(AttrDict):
    epsilon: float = 0.01
    rho0: float = 0.5
    omega: float = 0.5
    total_time_steps: int = 1000


lattice_grid_shape = (50, 50)


def density_equation(epsilon: float, omega: float) -> EquationFuncType:
    """ fourier equation (reference) """
    X, _ = lattice_grid_shape
    viscosity = 1. / 3. * (1. / omega - 0.5)

    def _imp(t: np.ndarray) -> np.ndarray:
        return epsilon * np.exp(-viscosity * (2 * np.pi / X) ** 2 * t)

    return _imp


def velocity_equation(epsilon: float, omega: float) -> EquationFuncType:
    _, Y = lattice_grid_shape
    viscosity = 1. / 3. * (1. / omega - 0.5)

    def _imp(t: np.ndarray) -> np.ndarray:
        return epsilon * np.exp(-viscosity * (2 * np.pi / Y) ** 2 * t)

    return _imp


def main(init_density: np.ndarray, init_velocity: np.ndarray,
         total_time_steps: int, omega: float, rho0: float, epsilon: float) -> None:
    X, Y = lattice_grid_shape
    field = FluidField2D(X, Y, omega=omega)
    field.init_vals(init_vel=init_velocity, init_density=init_density)

    densities = np.zeros(total_time_steps)
    vels = np.zeros(total_time_steps)
    field.local_equilibrium_pdf_update()

    for t in trange(total_time_steps):
        field.lattice_boltzmann_step()
        max_density = np.abs(field.density).max()
        max_vel = np.abs(field.velocity).max()
        densities[t] = max_density - rho0
        vels[t] = max_vel

    visualize_density_surface(field)
    for q, q_name, eq in [(densities, "density", density_equation(epsilon, omega)),
                          (vels, "velocity", velocity_equation(epsilon, omega))]:
        visualize_quantity_vs_time(
            quantities=q,
            quantity_name=q_name,
            equation=eq,
            total_time_steps=total_time_steps
        )


if __name__ == '__main__':
    kwargs = ExperimentVariables(
        epsilon=0.01,
        rho0=0.5,
        omega=1.95,
        total_time_steps=2000
    )
    density, vel = sinusoidal_density(lattice_grid_shape,
                                      epsilon=kwargs.epsilon,
                                      rho0=kwargs.rho0)
    main(init_density=density, init_velocity=vel, **kwargs)

    density, vel = sinusoidal_velocity(lattice_grid_shape,
                                       epsilon=kwargs.epsilon)
    main(init_density=density, init_velocity=vel, **kwargs)
