from typing import Optional, Tuple
from argparse import ArgumentParser

import numpy as np

from src.utils.utils import AttrDict, viscosity2omega
from src.experiments import (
    couette_flow_velocity_evolution,
    poiseuille_flow_velocity_evolution,
    sinusoidal_evolution,
    sinusoidal_viscosity,
    sliding_lid_seq,
    sliding_lid_mpi
)


class ExperimentVariables(AttrDict):
    total_time_steps: int
    lattice_grid_shape: Tuple[int, int]
    omega: float
    scaling_test: bool
    save: bool
    freq: int
    density_in: Optional[float]
    density_out: Optional[float]
    wall_vel: Optional[np.ndarray]


def run() -> None:
    name2func = {
        'cf': couette_flow_velocity_evolution,
        'pf': poiseuille_flow_velocity_evolution,
        'se': sinusoidal_evolution,
        'sv': sinusoidal_viscosity,
        'ss': sliding_lid_seq,
        'sm': sliding_lid_mpi
    }

    parser = ArgumentParser()
    parser.add_argument('-E', '--experiment', type=str, choices=name2func.keys(), required=True,
                        help=f'The experiment name. {name2func}.')
    parser.add_argument('-T', '--total_time_steps', type=int, required=True, help='The total time steps.')
    parser.add_argument('--freq', type=int, default=int(1e8), help='The frequency of saving data.')
    parser.add_argument('-X', type=int, required=True, help='The lattice size in the x direction.')
    parser.add_argument('-Y', type=int, required=True, help='The lattice size in the y direction.')
    parser.add_argument('--omega', type=float, help='The relaxation factor.')
    parser.add_argument('--visc', type=float, help='The viscosity.')
    parser.add_argument('-I', '--indensity', type=float, help='The density at the inlet.')
    parser.add_argument('-O', '--outdensity', type=float, help='The density at the outlet.')
    parser.add_argument('-W', '--wall_vel', type=float, help='The velocity of the wall along the x-axis.')
    parser.add_argument('--extrapolation', type=str, choices=['True', 'False'], default='False',
                        help='Whether using extrapolation for the wall density.')
    parser.add_argument('--eps', type=float, help='The amplitude of swinging in sinusoidal.')
    parser.add_argument('--rho', type=float, help='The offset of the density in sinusoidal..')
    parser.add_argument('--mode', type=str, choices=['d', 'v'], help='Either sinusoidal velocity or density.')
    parser.add_argument('--scaling', type=str, choices=['True', 'False'], default='False',
                        help='If performing scaling test.')
    parser.add_argument('--save', type=str, choices=['True', 'False'], default='True',
                        help='If save data or not.')

    args = parser.parse_args()
    if args.omega is None and args.visc is None:
        raise ValueError('Either omega or visc must be given from the argparser.')

    experiment_vars = ExperimentVariables(
        total_time_steps=args.total_time_steps,
        lattice_grid_shape=(args.X, args.Y),
        omega=args.omega if args.omega is not None else viscosity2omega(args.visc),
        scaling_test=eval(args.scaling),
        extrapolation=eval(args.extrapolation),
        save=eval(args.save),
        freq=args.freq
    )

    if args.indensity is not None and args.outdensity is not None:
        experiment_vars.update(
            density_in=args.indensity,
            density_out=args.outdensity
        )

    if args.wall_vel is not None:
        experiment_vars.update(wall_vel=np.array([args.wall_vel, 0.]))

    if args.mode is not None:
        mode2quant = {'v': 'velocity', 'd': 'density'}
        experiment_vars.update(
            mode=mode2quant[args.mode],
            rho0=args.rho,
            epsilon=args.eps
        )

    name2func[args.experiment](experiment_vars)


if __name__ == '__main__':
    run()
