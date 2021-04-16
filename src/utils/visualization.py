import matplotlib.pyplot as plt
import numpy as np

from src.utils.formula import FluidField2D


def visualize_velocity_streaming(field: FluidField2D):
    """
    Visualizie the velocity field as streaming
    """
    X, Y = field.lattice_grid_shape
    y, x = np.mgrid[:X, :Y]
    level = np.linalg.norm(field.velocity, axis=-1)
    plt.streamplot(x, y, field.velocity[..., 0], field.velocity[..., 1],
                   color=level, cmap='seismic')
    plt.xlim(0, X - 1)
    plt.ylim(0, Y - 1)
    plt.colorbar()
    plt.show()
