import numpy as np
from tqdm import trange
from typing import Tuple

from src.milestones.constants import sinusoidal_density, sinusoidal_velocity
from src.utils.visualization import visualize_density_surface, visualize_quantity_vs_time
from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.utils.attr_dict import AttrDict
from src.utils.constants import EquationFuncType


class ExperimentVariables(AttrDict):
    epsilon: float = 0.01
    rho0: float = 0.5
    omega: float = 0.5
    total_time_steps: int = 1000
    lattice_grid_shape: Tuple[int, int] = (50, 50)


def density_equation(epsilon: float, viscosity: float, lattice_grid_shape: Tuple[int, int]
                     ) -> EquationFuncType:
    """ fourier equation (reference) """
    X, _ = lattice_grid_shape

    def _imp(t: np.ndarray) -> np.ndarray:
        return epsilon * np.exp(-viscosity * (2 * np.pi / X) ** 2 * t)

    return _imp


def velocity_equation(epsilon: float, viscosity: float, lattice_grid_shape: Tuple[int, int]
                      ) -> EquationFuncType:
    _, Y = lattice_grid_shape

    def _imp(t: np.ndarray) -> np.ndarray:
        return epsilon * np.exp(-viscosity * (2 * np.pi / Y) ** 2 * t)

    return _imp


def main(init_density: np.ndarray, init_velocity: np.ndarray, lattice_grid_shape: Tuple[int, int],
         total_time_steps: int, omega: float, rho0: float, epsilon: float) -> None:

    X, Y = lattice_grid_shape
    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity, init_density=init_density)

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
    for q, q_name, eq in [(densities, "density", density_equation(epsilon, field.viscosity, lattice_grid_shape)),
                          (vels, "velocity", velocity_equation(epsilon, field.viscosity, lattice_grid_shape))]:
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
        total_time_steps=2000,
        lattice_grid_shape=(50, 50)
    )
    density, vel = sinusoidal_density(kwargs.lattice_grid_shape,
                                      epsilon=kwargs.epsilon,
                                      rho0=kwargs.rho0)
    main(init_density=density, init_velocity=vel, **kwargs)

    density, vel = sinusoidal_velocity(kwargs.lattice_grid_shape,
                                       epsilon=kwargs.epsilon)
    main(init_density=density, init_velocity=vel, **kwargs)
