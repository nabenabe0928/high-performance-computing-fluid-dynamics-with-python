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
    x, y = np.mgrid[:X, :Y]
    level = np.linalg.norm(field.velocity, axis=-1)
    plt.streamplot(x, y, field.velocity[..., 0], field.velocity[..., 1],
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
