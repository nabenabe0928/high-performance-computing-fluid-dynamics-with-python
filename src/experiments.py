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
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod
from src.simulation_attributes.boundary_handling import (
    MovingWall,
    PeriodicBoundaryConditions,
    RigidWall,
    sequential_boundary_handlings
)
from src.utils.utils import AttrDict, make_directories_to_path
from src.utils.constants import (
    DirectionIndicators,
    sinusoidal_density,
    sinusoidal_velocity,
    viscosity_equation
)
from src.utils.parallel_computation import ChunkedGridManager
from src.utils.utils import omega2viscosity
from src.utils.visualization import (
    PoiseuilleFlowHyperparams,
    visualize_couette_flow,
    visualize_density_plot,
    visualize_poiseuille_flow,
    visualize_velocity_plot,
    visualize_velocity_field
)


class ExperimentVariables(AttrDict):
    total_time_steps: int
    lattice_grid_shape: Tuple[int, int]
    init_density: np.ndarray
    init_velocity: np.ndarray
    omega: float
    scaling_test: bool
    save: bool
    epsilon: Optional[float]
    rho0: Optional[float]
    mode: Optional[str]
    in_density_factor: Optional[float]
    out_density_factor: Optional[float]
    wall_vel: Optional[np.ndarray]


def velocity0_density1(experiment_vars: ExperimentVariables, shape: Tuple[int, int]) -> None:
    experiment_vars.init_velocity = np.zeros((*shape, 2))
    experiment_vars.init_density = np.ones(shape)


def get_field(experiment_vars: ExperimentVariables, grid_manager: Optional[ChunkedGridManager] = None,
              dir_name: Optional[str] = None) -> LatticeBoltzmannMethod:

    field = LatticeBoltzmannMethod(
        *experiment_vars.lattice_grid_shape,
        omega=experiment_vars.omega,
        init_vel=experiment_vars.init_velocity,
        init_density=experiment_vars.init_density,
        grid_manager=grid_manager,
        dir_name=dir_name
    )

    return field


def sinusoidal_evolution(experiment_vars: ExperimentVariables, visualize: bool = True
                         ) -> Optional[float]:
    # Initialization
    lattice_grid_shape = experiment_vars.lattice_grid_shape
    mode = experiment_vars.mode
    save = experiment_vars.save

    if mode == 'density':
        eps, rho0 = experiment_vars.epsilon, experiment_vars.rho0

        assert eps is not None and rho0 is not None
        density, vel = sinusoidal_density(lattice_grid_shape,
                                          epsilon=eps,
                                          rho0=rho0)
        d_bounds = np.array([rho0 - eps, rho0 + eps])
        v_bounds = np.array([-eps, eps])
    elif mode == 'velocity':
        eps = experiment_vars.epsilon

        assert eps is not None
        density, vel = sinusoidal_velocity(lattice_grid_shape,
                                           epsilon=eps)
        d_bounds = np.array([-eps, eps])
        v_bounds = np.array([-eps, eps])

    experiment_vars.update(init_density=density, init_velocity=vel)
    total_time_steps = experiment_vars.total_time_steps
    subj = f'sinusoidal_{mode}'

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if save and visualize and (t == 0 or (t + 1) % 100 == 0):
            make_directories_to_path(f'log/{subj}/npy/')
            np.save(f'log/{subj}/npy/density{t + 1 if t else 0:0>6}.npy', field.density)
            np.save(f'log/{subj}/npy/v_abs{t + 1 if t else 0:0>6}.npy', field.velocity[..., 0])

    field = get_field(experiment_vars)
    # run LBM
    field(total_time_steps, proc=proc)
    if save and visualize:
        visualize_velocity_plot(subj, save=True, end=total_time_steps, bounds=v_bounds)
        visualize_density_plot(subj, save=True, end=total_time_steps, bounds=d_bounds)
        return None
    elif save:
        X, _ = field.lattice_grid_shape
        viscs = viscosity_equation(total_time_steps, eps, field.velocity[X // 2, :, 0])
        viscs = np.array([visc for visc in viscs if 0 < visc < 10])
        return float(np.mean(viscs))

    return None


def sinusoidal_viscosity(experiment_vars: ExperimentVariables) -> None:
    save = experiment_vars.save
    if not save:
        raise ValueError("sinusoidal_viscosity() runs only when save = True.")

    n_runs = 20
    visc_sim, visc_truth = [], []
    subj = f'sinusoidal_{experiment_vars.mode}'
    omegas = np.linspace(1e-6, 2 - 1e-6, n_runs + 1)[1:]
    for omega in omegas:
        experiment_vars.omega = omega
        v_sim = sinusoidal_evolution(experiment_vars, visualize=False)
        visc_truth.append(omega2viscosity(omega))
        visc_sim.append(v_sim)

    plt.plot(omegas, visc_truth, label="Analytical viscosity", color='red')
    plt.plot(omegas, visc_sim, label="Simulated result", color='blue')
    plt.scatter(omegas, visc_truth, marker='x', s=100, color='red')
    plt.scatter(omegas, visc_sim, marker='+', s=100, color='blue')
    plt.xlabel('$\\omega$')
    plt.ylabel('viscosity $\\nu$ (Log scale)')
    plt.yscale('log')
    plt.grid(which='both', color='black', linestyle='-')
    plt.legend(loc='lower left')
    plt.savefig(f'log/{subj}/fig/omega_vs_visc.pdf', bbox_inches='tight')


def couette_flow_velocity_evolution(experiment_vars: ExperimentVariables) -> None:
    save = experiment_vars.save

    # Initialization
    velocity0_density1(experiment_vars, experiment_vars.lattice_grid_shape)
    field = get_field(experiment_vars)
    total_time_steps = experiment_vars.total_time_steps
    rigid_wall = RigidWall(
        field=field,
        boundary_locations=[DirectionIndicators.TOP]
    )
    moving_wall = MovingWall(
        field=field,
        boundary_locations=[DirectionIndicators.BOTTOM],
        wall_vel=experiment_vars.wall_vel
    )

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if save and (t == 0 or (t + 1) % 100 == 0):
            make_directories_to_path('log/couette_flow/npy/')
            np.save(f'log/couette_flow/npy/v_x{t + 1 if t else 0 :0>6}.npy', field.velocity[..., 0])

    # run LBM
    field(total_time_steps, proc=proc, boundary_handling=sequential_boundary_handlings(rigid_wall, moving_wall))

    if save:
        visualize_couette_flow(wall_vel=experiment_vars.wall_vel, save=True, end=total_time_steps)


def poiseuille_flow_velocity_evolution(experiment_vars: ExperimentVariables) -> None:
    save = experiment_vars.save

    # Initialization
    velocity0_density1(experiment_vars, experiment_vars.lattice_grid_shape)
    field = get_field(experiment_vars)
    total_time_steps = experiment_vars.total_time_steps

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

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if save and (t == 0 or (t + 1) % 100 == 0):
            make_directories_to_path('log/poiseuille_flow/npy/')
            np.save(f'log/poiseuille_flow/npy/v_x{t + 1 if t else 0:0>6}.npy', field.velocity[..., 0])
            np.save(f'log/poiseuille_flow/npy/density{t + 1 if t else 0:0>6}.npy', field.density)

    # run LBM
    field(total_time_steps, proc=proc, boundary_handling=sequential_boundary_handlings(rigid_wall, pbc))
    params = PoiseuilleFlowHyperparams(
        viscosity=omega2viscosity(experiment_vars.omega),
        out_density_factor=experiment_vars.out_density_factor,
        in_density_factor=experiment_vars.in_density_factor
    )

    if save:
        visualize_poiseuille_flow(params, save=True, end=total_time_steps)


def sliding_lid_seq(experiment_vars: ExperimentVariables) -> None:
    save = experiment_vars.save

    # Initialization
    velocity0_density1(experiment_vars, experiment_vars.lattice_grid_shape)
    X, Y = experiment_vars.lattice_grid_shape
    visc = omega2viscosity(experiment_vars.omega)

    assert experiment_vars.wall_vel is not None
    dir_name = f'sliding_lid_W{experiment_vars.wall_vel[0]:.2f}_visc{visc:.2f}_size{X}'

    field = get_field(experiment_vars, dir_name=dir_name)
    total_time_steps = experiment_vars.total_time_steps

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

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if save and (t == 0 or (t + 1) % 100 == 0):
            path = f'log/{dir_name}/npy/'
            make_directories_to_path(path)
            np.save(f'{path}v_abs{t + 1 if t else 0:0>6}.npy', np.linalg.norm(field.velocity, axis=-1))
            np.save(f'{path}v_x{t + 1 if t else 0:0>6}.npy', field.velocity[..., 0])
            np.save(f'{path}v_y{t + 1 if t else 0:0>6}.npy', field.velocity[..., 1])

    # run LBM
    field(total_time_steps, proc=proc, boundary_handling=sequential_boundary_handlings(rigid_wall, moving_wall))

    if save:
        visualize_velocity_field(subj=dir_name, save=True, end=total_time_steps)


def sliding_lid_mpi(experiment_vars: ExperimentVariables) -> None:
    save = experiment_vars.save

    # Initialization
    scaling = experiment_vars.scaling_test
    grid_manager = ChunkedGridManager(*experiment_vars.lattice_grid_shape)
    velocity0_density1(experiment_vars, grid_manager.buffer_grid_size)
    X, Y = experiment_vars.lattice_grid_shape
    visc = omega2viscosity(experiment_vars.omega)

    assert experiment_vars.wall_vel is not None
    dir_name = f'sliding_lid_W{experiment_vars.wall_vel[0]:.2f}_visc{visc:.2f}_size{X}'

    field = get_field(experiment_vars, grid_manager=grid_manager, dir_name=dir_name)
    start, total_time_steps = time.time(), experiment_vars.total_time_steps

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

    freq = 5000

    def proc(field: LatticeBoltzmannMethod, t: int) -> None:
        if save and not scaling and (t + 1) % freq == 0:
            field.save_velocity_field(t + 1 if t else 0)

    # run LBM
    field(total_time_steps, proc=proc, boundary_handling=sequential_boundary_handlings(rigid_wall, moving_wall))

    if save and not scaling:
        visualize_velocity_field(dir_name, save=True, start=freq, end=total_time_steps + 1, freq=freq)

    if scaling and grid_manager.rank == 0:
        end = time.time()
        X, Y = experiment_vars.lattice_grid_shape
        MLUPS = X * Y * experiment_vars.total_time_steps / (end - start)
        path = f'log/{dir_name}/'
        make_directories_to_path(path)
        with open(f'log/{dir_name}/MLUPS_vs_proc.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([grid_manager.size, MLUPS])
