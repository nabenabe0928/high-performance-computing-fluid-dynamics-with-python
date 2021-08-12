import numpy as np
from typing import Tuple

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import (
    PeriodicBoundaryConditionsWithPressureVariation,
    RigidWall,
    SequentialBoundaryHandlings
)
from src.utils.utils import AttrDict
from src.utils.constants import DirectionIndicators
# from src.utils.visualization import visualize_poiseuille_flow


class ExperimentVariables(AttrDict):
    omega: float = 0.5
    total_time_steps: int = 1000
    density_in: float = 1.0 + 0.005
    density_out: float = 1.0 - 0.005
    lattice_grid_shape: Tuple[int, int] = (30, 30)


def main(init_density: np.ndarray, init_velocity: np.ndarray,
         total_time_steps: int, omega: float, lattice_grid_shape: Tuple[int, int],
         density_in: float, density_out: float) -> None:

    X, Y = lattice_grid_shape

    field = LatticeBoltzmannMethod(X, Y, omega=omega, init_vel=init_velocity, init_density=init_density)

    pbc = PeriodicBoundaryConditionsWithPressureVariation(
        field=field,
        boundary_locations=[DirectionIndicators.LEFT, DirectionIndicators.RIGHT],
        density_in=density_in,
        density_out=density_out
    )

    rigid_wall = RigidWall(
        field=field,
        boundary_locations=[DirectionIndicators.TOP, DirectionIndicators.BOTTOM]
    )

    field(total_time_steps, boundary_handling=SequentialBoundaryHandlings(rigid_wall, pbc))
    # visualize_poiseuille_flow(field=field, pbc=pbc)


if __name__ == '__main__':
    kwargs = ExperimentVariables(
        omega=1.5,
        total_time_steps=5000,
        lattice_grid_shape=(30, 30)
    )
    X, Y = kwargs.lattice_grid_shape

    density, vel = np.ones((X, Y)), np.zeros((X, Y, 2))
    main(init_density=density, init_velocity=vel, **kwargs)
