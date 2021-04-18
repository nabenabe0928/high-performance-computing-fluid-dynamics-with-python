import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.signal import argrelextrema

from src.simulation_attributes.formula import FluidField2D
from src.utils.constants import EquationFuncType


DEFAULT_CMAP = 'gist_rainbow'


def visualize_velocity_streaming(field: FluidField2D, cmap: str = DEFAULT_CMAP) -> None:
    """
    Visualizie the velocity field as streaming
    """
    X, Y = field.lattice_grid_shape
    y, x = np.mgrid[:X, :Y]
    # since when v(t) = 0, it raises error, add the buffer
    vel = field.velocity + 1e-12

    level = np.linalg.norm(field.velocity, axis=-1)

    plt.streamplot(x, y, vel[..., 0], vel[..., 1],
                   color=level, cmap='gist_rainbow')
    plt.xlim(0, X - 1)
    plt.ylim(0, Y - 1)
    plt.colorbar()
    plt.show()


def visualize_density_surface(field: FluidField2D, cmap: str = DEFAULT_CMAP) -> None:
    X, Y = field.lattice_grid_shape
    x, y = np.mgrid[:X, :Y]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, field.density, cmap=cmap)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(field.density)
    plt.title('Surface plot of density at each location')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(m)
    plt.show()


def visualize_quantity_vs_time(quantities: np.ndarray, quantity_name: str,
                               total_time_steps: int, equation: EquationFuncType
                               ) -> None:
    indices = argrelextrema(quantities, np.greater)[0]
    extrema = quantities[indices]

    t = np.arange(total_time_steps)
    analytical_vals = equation(t)

    plt.plot(indices, extrema, label=f"Simulated cumulated max {quantity_name}")
    plt.plot(t, quantities, label=f"Simulated {quantity_name}")
    plt.plot(t, analytical_vals, label=f"Analytical {quantity_name}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(f"Amplitude of {quantity_name}")
    plt.show()


def visualize_velocity_field_of_moving_wall(field: FluidField2D, wall_vel: np.ndarray) -> None:
    """ we assume the wall slides to x-direction. """
    vx = field.velocity[..., 0]
    X, Y = field.lattice_grid_shape
    wv = wall_vel[0]
    assert wall_vel[1] == 0

    for vxy, y in zip(vx[X // 2, :], np.arange(Y)):
        src = [0, y]
        arrow = [vxy, 0]
        plt.quiver(*src, *arrow, color='red', scale_units='xy', scale=1, headwidth=3, width=3e-3)

    print(vx)
    plt.plot(vx[X // 2, :], np.arange(Y), label="Simulated result", color="blue", linestyle=":", linewidth=1)
    plt.plot(wv * (Y - np.arange(Y + 1)) / Y, np.arange(Y + 1) - 0.5, label="Theoretical value")

    vmax = int(max(wv, np.ceil(vx[X // 2, :].max()))) + 1
    plt.plot(np.arange(vmax), np.ones(vmax) * (Y - 1) + 0.5, label="Rigid wall")
    plt.plot(np.arange(vmax), np.zeros(vmax) - 0.5, label='Moving wall')

    plt.ylabel('y axis')
    plt.xlabel('Velocity in y-axis')
    plt.legend()
    plt.show()
