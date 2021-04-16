import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from src.utils.formula import FluidField2D


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
