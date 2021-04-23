"""
# What I have to put on the final reports
## Milestone 3
@ 1. The evolution of the density and the velocity profiles over time
x 2. The measured viscosity v.s. the parameter omega

## Milestone 4 (Couette Flow)
@ 1. The evolution of the velocity profile over time

## Milestone 5 (Poiseuille Flow)
@ 1. The evolution of the velocity profile over time
@ 2. The comparison between the analytical solutions and the results

## Milestone 6 (The sliding lid)
@ 1. Observe the results with a fixed box size and Reynolds number 1000.

## Milestone 7 (MPI)
@ 1. The comparison between Million Lattice Updates Per Second (MLUPS) v.s. runtime
@ 2. The comparison of the scaling between different lattice size

NOTE: one lattice == one point
"""

import csv
import numpy as np
import time
from tqdm import trange
from typing import Callable, Optional, Tuple

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import (
    MovingWall,
    PeriodicBoundaryConditions,
    RigidWall
)
from src.utils.attr_dict import AttrDict
from src.utils.constants import (
    DirectionIndicators,
    density_equation,
    sinusoidal_density,
    sinusoidal_velocity,
    velocity_equation
)
from src.utils.parallel_computation import ChunkedGridManager
from src.utils.visualization import (
    visualize_density_surface,
    visualize_quantity_vs_time,
    visualize_velocity_field_of_moving_wall,
    visualize_velocity_field_of_pipe,
    visualize_velocity_field,
    visualize_velocity_field_mpi
)


class ExperimentVariables(AttrDict):
    total_time_steps: int
    lattice_grid_shape: Tuple[int, int]
    init_density: np.ndarray
    init_velocity: np.ndarray
    omega: float
    epsilon: Optional[float]
    rho0: Optional[float]
    in_density_factor: Optional[float]
    out_density_factor: Optional[float]
    wall_vel: Optional[np.ndarray]


BoundaryHandlingFuncType = Callable[[LatticeBoltzmannMethod], None]
ProcessFuncType = Callable[[LatticeBoltzmannMethod, int], None]


def get_field(experiment_vars: ExperimentVariables) -> LatticeBoltzmannMethod:
    X, Y = experiment_vars.lattice_grid_shape
    field = LatticeBoltzmannMethod(
        X, Y,
        omega=experiment_vars.omega,
        init_vel=experiment_vars.init_velocity,
        init_density=experiment_vars.init_density
    )

    return field


def run_experiment(field: LatticeBoltzmannMethod,
                   experiment_vars: ExperimentVariables,
                   proc: Optional[ProcessFuncType] = None,
                   boundary_handling: Optional[BoundaryHandlingFuncType] = None
                   ) -> None:

    field.local_equilibrium_pdf_update()
    for t in trange(experiment_vars.total_time_steps):
        field.lattice_boltzmann_step(boundary_handling=boundary_handling)
        if proc is not None:
            proc(field, t)


def density_and_velocity_evolution(experiment_vars: ExperimentVariables) -> None:
    lattice_grid_shape = experiment_vars.lattice_grid_shape
    total_time_steps = experiment_vars.total_time_steps

    densities, vels = np.zeros(total_time_steps), np.zeros(total_time_steps)

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        densities[t] = np.abs(field.density).max() - experiment_vars.rho0
        vels[t] = np.abs(field.velocity).max()

    field = get_field(experiment_vars)
    run_experiment(field, experiment_vars, proc)

    visualize_density_surface(field)
    eps, visc = experiment_vars.epsilon, field.viscosity
    for q, q_name, eq in [(densities, "density", density_equation(eps, visc, lattice_grid_shape)),
                          (vels, "velocity", velocity_equation(eps, visc, lattice_grid_shape))]:
        visualize_quantity_vs_time(
            quantities=q,
            quantity_name=q_name,
            equation=eq,
            total_time_steps=total_time_steps
        )


def couette_flow_velocity_evolution(experiment_vars: ExperimentVariables) -> None:
    experiment_vars.init_velocity = np.zeros((*experiment_vars.lattice_grid_shape, 2))
    experiment_vars.init_density = np.ones(experiment_vars.lattice_grid_shape)
    field = get_field(experiment_vars)
    wall_vel = experiment_vars.wall_vel
    rigid_wall = RigidWall(
        field=field,
        boundary_locations=[DirectionIndicators.TOP]
    )
    moving_wall = MovingWall(
        field=field,
        boundary_locations=[DirectionIndicators.BOTTOM],
        wall_vel=wall_vel
    )

    def boundary_handling(field: LatticeBoltzmannMethod) -> None:
        rigid_wall.boundary_handling(field)
        moving_wall.boundary_handling(field)

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if (t + 1) % 100 == 0:
            visualize_velocity_field_of_moving_wall(field=field, wall_vel=wall_vel)

    run_experiment(field, experiment_vars, proc, boundary_handling)


def poiseuille_flow_velocity_evolution(experiment_vars: ExperimentVariables) -> None:
    experiment_vars.init_velocity = np.zeros((*experiment_vars.lattice_grid_shape, 2))
    experiment_vars.init_density = np.ones(experiment_vars.lattice_grid_shape)
    field = get_field(experiment_vars)

    pbc = PeriodicBoundaryConditions(
        field=field,
        boundary_locations=[DirectionIndicators.LEFT, DirectionIndicators.RIGHT],
        in_density_factor=experiment_vars.in_density_factor,
        out_density_factor=experiment_vars.out_density_factor
    )

    rigid_wall = RigidWall(
        field=field,
        boundary_locations=[DirectionIndicators.TOP, DirectionIndicators.BOTTOM]
    )

    def boundary_handling(field: LatticeBoltzmannMethod) -> None:
        pbc.boundary_handling(field)
        rigid_wall.boundary_handling(field)

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if (t + 1) % 100 == 0:
            visualize_velocity_field_of_pipe(field=field, pbc=pbc)

    run_experiment(field, experiment_vars, proc, boundary_handling)


def sliding_lid_velocity_evolution_seq(experiment_vars: ExperimentVariables) -> None:
    experiment_vars.init_velocity = np.zeros((*experiment_vars.lattice_grid_shape, 2))
    experiment_vars.init_density = np.ones(experiment_vars.lattice_grid_shape)
    field = get_field(experiment_vars)

    moving_wall = MovingWall(
        field,
        boundary_locations=[DirectionIndicators.BOTTOM],
        wall_vel=experiment_vars.wall_vel
    )

    rigid_wall = RigidWall(
        field,
        boundary_locations=[
            DirectionIndicators.TOP,
            DirectionIndicators.LEFT,
            DirectionIndicators.RIGHT
        ]
    )

    def boundary_handling(field: LatticeBoltzmannMethod) -> None:
        rigid_wall.boundary_handling(field)
        moving_wall.boundary_handling(field)

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if (t + 1) % 100 == 0:
            visualize_velocity_field(field=field)

    run_experiment(field, experiment_vars, proc, boundary_handling)


def sliding_lid_mpi(experiment_vars: ExperimentVariables,
                    scaling: bool = False) -> None:

    X, Y = experiment_vars.lattice_grid_shape
    grid_manager = ChunkedGridManager(X, Y)
    buffer_grid_size = grid_manager.buffer_grid_size
    experiment_vars.init_velocity = np.zeros((*buffer_grid_size, 2))
    experiment_vars.init_density = np.ones(buffer_grid_size)
    field = get_field(experiment_vars)
    start = time.time_ns()

    rigid_boundary_locations = [
        getattr(DirectionIndicators, dir)
        for dir in ['TOP', 'LEFT', 'RIGHT']
        if grid_manager.is_boundary(getattr(DirectionIndicators, dir))
    ]

    moving_wall, rigid_wall = None, None
    if grid_manager.is_boundary(DirectionIndicators.BOTTOM):
        moving_wall = MovingWall(
            field,
            boundary_locations=[DirectionIndicators.BOTTOM],
            wall_vel=experiment_vars.wall_vel
        )
    if len(rigid_boundary_locations) >= 1:
        rigid_wall = RigidWall(
            field,
            boundary_locations=rigid_boundary_locations
        )

    def boundary_handling(field: LatticeBoltzmannMethod) -> None:
        if rigid_wall is not None:
            rigid_wall.boundary_handling(field)
        if moving_wall is not None:
            moving_wall.boundary_handling(field)

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if not scaling:
            if (t + 1) % 100 == 0:
                x_file, y_file = field.save_velocity_field(
                    dir_name='test_run',
                    file_name='v',
                    index=t + 1
                )
                if field.grid_manager.rank == 0:
                    visualize_velocity_field_mpi(x_file, y_file)

    run_experiment(field, experiment_vars, proc, boundary_handling)

    if scaling and grid_manager.rank == 0:
        end = time.time_ns()
        runtime = (end - start) / 1e9
        MLUPS = X * Y * experiment_vars.total_time_steps / runtime
