import numpy as np
from tqdm import trange

from src.utils.visualization import visualize_velocity_streaming
from src.simulation_attributes.formula import FluidField2D


def main() -> None:
    X, Y = 50, 50
    total_time_steps = 20

    pdf = np.zeros((X, Y, 9))
    pdf[:X//2, :Y//2, 5] = np.ones((X // 2, Y // 2))
    field = FluidField2D(X, Y)
    field.init_vals(init_pdf=pdf)

    for t in trange(total_time_steps):
        field.update_density()
        field.update_velocity()
        field.update_pdf()
        visualize_velocity_streaming(field)


if __name__ == '__main__':
    main()
