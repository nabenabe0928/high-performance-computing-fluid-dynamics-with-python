from argparse import ArgumentParser

from src.utils.utils import compare_serial_vs_parallel


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-T', '--total_time_steps', type=int, required=True, help='The total time steps.')
    parser.add_argument('-X', type=int, required=True, help='The lattice size in the x direction.')
    parser.add_argument('-Y', type=int, required=True, help='The lattice size in the y direction.')
    parser.add_argument('--visc', type=float, help='The viscosity.')
    parser.add_argument('-W', '--wall_vel', type=float, help='The velocity of the wall along the x-axis.')

    args = parser.parse_args()

    compare_serial_vs_parallel(wall_vel=args.wall_vel, viscosity=args.visc,
                               X=args.X, Y=args.Y, T=args.total_time_steps)
