import numpy as np

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod


def main() -> None:
    X, Y = 50, 50
    total_time_steps = 20

    pdf = np.zeros((X, Y, 9))
    pdf[:X // 2, :Y // 2, 5] = np.ones((X // 2, Y // 2))
    field = LatticeBoltzmannMethod(X, Y, init_pdf=pdf)

    # for t in trange(total_time_steps):
    for t in range(total_time_steps):
        field.update_density()
        field.update_velocity()
        field.update_pdf()
        # visualize_velocity_field(field)


if __name__ == '__main__':
    main()
