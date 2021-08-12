import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from typing import NamedTuple, Optional

from src.utils.constants import EquationFuncType
from src.utils.utils import make_directories_to_path

# This import is for the 3D plot (if you remove, you will yield an error.)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


plt.rcParams['mathtext.fontset'] = 'stix' # The setting of math font


class PoiseuilleFlowHyperparams(NamedTuple):
    viscosity: float
    density_in: float
    density_out: float


def show_or_save(path: Optional[str] = None) -> None:
    if path is None:
        plt.show()
    else:
        file_name = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])
        format = file_name.split('.')[-1]
        if format not in ['pdf', 'png']:
            raise ValueError(f'The format must be either pdf or png, but got {format}.')

        make_directories_to_path(path)
        plt.savefig(f'{path}/{file_name}', bbox_inches='tight')


def simulated_arrows(vx: np.ndarray, X: int, Y: int, v_analy: np.ndarray) -> None:
    for vxy, y in zip(vx[X // 2, :], np.arange(Y)):
        src, arrow = [0, y], [vxy, 0]
        plt.quiver(*src, *arrow, color='red', scale_units='xy', scale=1, headwidth=3, width=3e-3)

    plt.plot(vx[X // 2, :], np.arange(Y), label="Simulated result", color="blue", linestyle=":", linewidth=2)
    plt.plot(v_analy, np.arange(Y + 1) - 0.5, label="Analytical velocity")
    plt.ylabel('y axis')
    plt.xlabel('velocity in y axis')


def visualize_velocity_countour(subject: str, save: bool = False, format: str = 'pdf', start: int = 0,
                                freq: int = 100, end: int = 100001, cmap: Optional[str] = None,
                                bounds: Optional[np.ndarray] = None) -> None:

    levels: Optional[np.ndarray] = None
    if bounds is not None:
        assert bounds.shape == (2,)
        levels = np.linspace(bounds[0], bounds[1], 100)

    for t in range(start, end, freq):
        v_abs_file_name = f'log/{subject}/npy/v_abs{t:0>6}.npy'
        v = np.load(v_abs_file_name)
        X, Y = v.shape
        x, y = np.meshgrid(np.arange(X), np.arange(Y))

        plt.close('all')
        plt.figure()
        plt.xlim(0, X)
        plt.ylim(0, Y)
        plt.contourf(x, y, v, cmap=cmap, levels=levels)
        show_or_save(path=f'log/{subject}/fig/vel{t:0>6}.{format}' if save else None)


def visualize_vel_rot_countour(subject: str, save: bool = False, format: str = 'pdf', start: int = 0,
                               freq: int = 100, end: int = 100001, cmap: Optional[str] = None) -> None:

    for t in range(start, end, freq):
        x_file_name, y_file_name = f'log/{subject}/npy/v_x{t:0>6}.npy', f'log/{subject}/npy/v_y{t:0>6}.npy'
        vx, vy = np.load(x_file_name), np.load(y_file_name)
        deriv_vx_y = (np.roll(vx, shift=-1, axis=1) - np.roll(vx, shift=1, axis=1)) / 2
        deriv_vy_x = (np.roll(vy, shift=-1, axis=0) - np.roll(vy, shift=1, axis=0)) / 2
        rot_amp = np.abs(deriv_vy_x[1:-1, 1:-1] - deriv_vx_y[1:-1, 1:-1])
        rot_amp /= np.amax(rot_amp)
        rot_amp = np.log(rot_amp + 1e-12)

        X, Y = rot_amp.shape
        x, y = np.meshgrid(np.arange(X), np.arange(Y))

        plt.close('all')
        plt.figure()
        plt.xlim(0, X)
        plt.ylim(0, Y)
        plt.contourf(x, y, rot_amp, cmap=cmap, levels=np.linspace(np.log(1e-12), 0., 100))
        show_or_save(path=f'log/{subject}/fig/rot_contour{t:0>6}.{format}' if save else None)


def visualize_density_countour(subject: str, save: bool = False, format: str = 'pdf', start: int = 0,
                               freq: int = 100, end: int = 100001, cmap: Optional[str] = None,
                               bounds: Optional[np.ndarray] = None) -> None:

    levels: Optional[np.ndarray] = None
    if bounds is not None:
        assert bounds.shape == (2,)
        levels = np.linspace(bounds[0], bounds[1], 100)

    for t in range(start, end, freq):
        density_file_name = f'log/{subject}/npy/density{t:0>6}.npy'
        density = np.load(density_file_name)
        # density /= np.amax(density)
        X, Y = density.shape
        x, y = np.meshgrid(np.arange(X), np.arange(Y))

        plt.close('all')
        plt.figure()
        plt.xlim(0, X)
        plt.ylim(0, Y)
        plt.contourf(x, y, density, levels=levels, cmap=cmap)
        show_or_save(path=f'log/{subject}/fig/density{t:0>6}.{format}' if save else None)


def visualize_density_plot(subject: str, profile: np.ndarray, save: bool = False, format: str = 'pdf',
                           start: int = 0, freq: int = 100, end: int = 100001,
                           cmap: Optional[str] = None, bounds: np.ndarray = np.array([0., 1.])) -> None:

    buf = (bounds[1] - bounds[0]) * 0.1
    for t in range(start, end, freq):
        density_file_name = f'log/{subject}/npy/density{t:0>6}.npy'
        density = np.load(density_file_name)
        X, Y = density.shape

        plt.close('all')
        plt.figure(figsize=(5, 3))
        plt.ylabel(f'Density $\\rho(x, y={Y//2})$')
        plt.xlabel(f'$x$ position')

        plt.xlim(0, X)
        plt.ylim(bounds[0] - buf, bounds[1] + buf)
        plt.plot(np.arange(X), density[:, Y // 2], label='Simulated Result')

        if t == 0:
            plt.legend()

        show_or_save(path=f'log/{subject}/fig/density{t:0>6}.{format}' if save else None)

    plt.close('all')
    plt.figure(figsize=(20, 3))
    T = np.arange(profile.size)
    plt.plot(T, profile, label='Simulated Result')
    plt.xlabel('Time step $t$')
    plt.ylabel('Density $\\rho(x={}, y={})$'.format(X//4, Y//2))
    dy = profile.max() - profile.min()
    plt.ylim(profile.min() - dy * 0.1, profile.max() + dy * 0.1)
    plt.grid()
    plt.legend()
    show_or_save(path=f'log/{subject}/fig/density_time_evolution.{format}' if save else None)


def visualize_velocity_plot(subject: str, profile: np.ndarray, epsilon: float, visc: float,
                            save: bool = False, format: str = 'pdf', start: int = 0, freq: int = 100,
                            end: int = 100001, cmap: Optional[str] = None,
                            bounds: Optional[np.ndarray] = None) -> None:

    assert bounds is not None
    buf = (bounds[1] - bounds[0]) * 0.1

    def analytical_sol(t: int, y: np.ndarray, Y: int) -> float:
        ly = 2.0 * np.pi / Y
        return epsilon * np.exp(- visc * ly ** 2 * t) * np.sin(ly * y)

    for t in range(start, end, freq):
        v_abs_file_name = f'log/{subject}/npy/v_abs{t:0>6}.npy'
        v = np.load(v_abs_file_name)
        X, Y = v.shape
        y = np.arange(Y)

        plt.close('all')
        plt.figure(figsize=(5, 3))
        plt.xlim(0, Y)
        plt.ylim(bounds[0] - buf, bounds[1] + buf)
        plt.plot(y, analytical_sol(t, y, Y), color='blue', lw=4.5, label='Analytical Solution')
        plt.plot(y, v[X // 2, :], color='red', label='Simulated Result')
        if t == 0:
            plt.legend()

        plt.ylabel('Velocity $u_x(x={}, y)$'.format(X//2))
        plt.xlabel('$y$ position')

        show_or_save(path=f'log/{subject}/fig/vel{t:0>6}.{format}' if save else None)

    plt.close('all')
    plt.figure(figsize=(20, 3))
    T = np.arange(profile.size)
    plt.plot(T, analytical_sol(T, Y//4, Y), color='blue', lw=4.5, label='Analytical Solution')
    plt.plot(T, profile, color='red', label='Simulated Result')
    plt.xlabel('Time step $t$')
    plt.ylabel(f'Velocity $u_x(y={Y//4})$')
    plt.grid()
    plt.legend()
    show_or_save(path=f'log/{subject}/fig/vel_time_evolution.{format}' if save else None)


def visualize_velocity_field(subj: str, save: bool = False, format: str = 'pdf', start: int = 0,
                             freq: int = 100, end: int = 100001, cmap: Optional[str] = None) -> None:

    """ sinusoidal or sliding_lid """
    if subj[:11] != 'sliding_lid' and subj != 'sinusoidal':
        raise ValueError(f'subj must be either sliding_lid or sinusoidal, but got {subj}.')

    for t in range(start, end, freq):
        vx_file_name, vy_file_name = f'log/{subj}/npy/v_x{t:0>6}.npy', f'log/{subj}/npy/v_y{t:0>6}.npy'
        vx, vy = np.load(vx_file_name), np.load(vy_file_name)
        X, Y = vx.shape
        x, y = np.meshgrid(np.arange(X), np.arange(Y))

        plt.close('all')
        plt.figure()
        plt.xlim(0, X)
        plt.xlabel('$x$ position')
        plt.ylabel('$y$ position')
        plt.ylim(0, Y)
        level = np.linalg.norm(np.dstack([vy, vx]), axis=-1).T
        plt.streamplot(x, y, vx.T, vy.T, color=level, cmap=cmap)
        show_or_save(path=f'log/{subj}/fig/vel_flow{t:0>6}.{format}' if save else None)


def visualize_couette_flow(wall_vel: np.ndarray, save: bool = False, format: str = 'pdf', start: int = 0,
                           freq: int = 400, end: int = 100001, cmap: Optional[str] = None, joint: bool = True) -> None:
    """ we assume the wall slides to x-direction. """
    plt.figure(figsize=(5, 3))
    t_last = 0

    for t in range(start, end, freq):
        t_last = t
        vx_file_name = f'log/couette_flow/npy/v_x{t:0>6}.npy'
        vx = np.load(vx_file_name)
        X, Y = vx.shape
        analy_sol = wall_vel[0] * (Y - np.arange(Y + 1)) / Y
        vmax = int(max(wall_vel[0], np.ceil(vx[X // 2, :].max()))) + 1

        if not joint:
            plt.close('all')
            plt.figure(figsize=(5, 3))
            plt.xlim(-0.01 * wall_vel[0], wall_vel[0])
            plt.ylim(-0.5, Y)
            simulated_arrows(vx, X, Y, analy_sol)
            plt.plot(np.arange(vmax), np.ones(vmax) * (Y - 1) + 0.5, label="Rigid wall", color='black', linewidth=3.0)
            plt.plot(np.arange(vmax), np.zeros(vmax), label='Moving wall', color='blue', linewidth=3.0)

            if t == start:
                plt.rc('legend', fontsize=16)
                plt.legend(loc='upper right')

            show_or_save(path=f'log/couette_flow/fig/couette_flow{t:0>6}.{format}' if save else None)
        else:
            if t == start:
                plt.plot(np.arange(vmax), np.ones(vmax) * (Y - 1) + 0.5, label="Rigid wall",
                         color='black', linewidth=3.0)
                plt.plot(np.arange(vmax), np.zeros(vmax), label='Moving wall', color='red', linewidth=3.0)
                plt.xlim(-0.01 * wall_vel[0], wall_vel[0])
                plt.ylim(-0.5, Y)
                plt.plot(analy_sol, np.arange(Y + 1) - 0.5, color='blue', lw=4.5, label="Analytical solution")

            plt.plot(vx[X // 2, :], np.arange(Y), linestyle=":", linewidth=2)

    if joint:
        dx = analy_sol[0]
        plt.xlim(-dx * 0.2, dx * 1.1)
        plt.annotate(
            s='$t = {}$'.format(t_last),
            xy=[analy_sol[Y//4], Y//4],
            xytext=[analy_sol[Y//3] + dx * 0.08, Y//3],
            arrowprops=dict(
                facecolor='red',
                edgecolor='red',
                arrowstyle = '->'
            )
        )
        plt.annotate(
            s='$t = 0$',
            xy=[0, Y-Y//4],
            xytext=[-dx*0.17, Y-Y//8],
            arrowprops=dict(
                facecolor='red',
                edgecolor='red',
                arrowstyle = '->'
            )
        )
        plt.rc('legend', fontsize=12)
        plt.xlabel(f'Velocity $u_x(x={X//2}, y)$')
        plt.ylabel('$y$ position')
        plt.legend(loc='upper right')
        show_or_save(path=f'log/couette_flow/fig/couette_flow_joint.{format}' if save else None)


def visualize_poiseuille_flow(params: PoiseuilleFlowHyperparams, save: bool = False, format: str = 'pdf',
                              start: int = 0, freq: int = 1000, end: int = 100001, cmap: Optional[str] = None,
                              joint: bool = True) -> None:

    """ we assume the wall slides to x-direction. """
    viscosity = params.viscosity
    density_out = params.density_out
    density_in = params.density_in
    t_last = 0
    V = []

    for t in range(start, end, freq):
        t_last = t
        vx_file_name = f'log/poiseuille_flow/npy/v_x{t:0>6}.npy'
        density_file_name = f'log/poiseuille_flow/npy/density{t:0>6}.npy'
        vx, density = np.load(vx_file_name), np.load(density_file_name)
        X, Y = density.shape
        _, y = np.meshgrid(np.arange(X), np.arange(Y))
        average_density = viscosity * density[X // 2, :].mean()
        # density / 3.0 = pressure
        deriv_density_x = (density_out - density_in) / X / 3.0
        y = np.arange(Y + 1)
        analy_sol = - 0.5 * deriv_density_x * y * (Y - y) / average_density

        if not joint:
            plt.close('all')
            plt.figure(figsize=(5, 3))
            plt.xlim(0, 0.023)
            plt.ylim(-0.5, Y)
            simulated_arrows(vx, X, Y, analy_sol)

            if t == start:
                plt.rc('legend', fontsize=16)
                plt.legend(loc='upper right')

            show_or_save(path=f'log/poiseuille_flow/fig/poiseuille_flow{t:0>6}.{format}' if save else None)
        else:
            if t == start:
                plt.figure(figsize=(5, 3))
                plt.ylim(-0.5, Y)

            V.append(vx[X // 2, :])

    if joint:
        plt.xlim(0, analy_sol.max() * 1.1)
        plt.plot(analy_sol, np.arange(Y + 1) - 0.5, color='blue', lw=4.5, label="Analytical solution")
        plt.rc('legend', fontsize=12)
        idx = 0
        y = np.arange(V[0].size)
        for t in range(start, end, freq):
            plt.plot(V[idx], y, linestyle=":", linewidth=2)
            idx += 1

        dx = analy_sol[Y//2]
        plt.xlim(-dx * 0.2, dx * 1.1)
        plt.annotate(
            s='$t = {}$'.format(t_last),
            xy=[analy_sol[-Y//4], Y-Y//4],
            xytext=[analy_sol[-Y//8] + dx * 0.2, Y - Y//8],
            arrowprops=dict(
                facecolor='red',
                edgecolor='red',
                arrowstyle = '->'
            )
        )
        plt.annotate(
            s='$t = 0$',
            xy=[0, Y-Y//4],
            xytext=[-dx*0.17, Y-Y//8],
            arrowprops=dict(
                facecolor='red',
                edgecolor='red',
                arrowstyle = '->'
            )
        )
        plt.xlabel(f'Velocity $u_x(x={X//2}, y)$')
        plt.ylabel('$y$ position')
        plt.legend(loc='lower right')
        show_or_save(path=f'log/poiseuille_flow/fig/poiseuille_flow_joint.{format}' if save else None)


def visualize_quantity_vs_time(quantities: np.ndarray, quantity_name: str,
                               total_time_steps: int,
                               equation: Optional[EquationFuncType] = None
                               ) -> None:
    indices = argrelextrema(quantities, np.greater)[0]
    extrema = quantities[indices]

    t = np.arange(total_time_steps)
    if equation is not None:
        analytical_vals = equation(t)
        plt.plot(t, analytical_vals, label=f"Analytical {quantity_name}")

    plt.plot(indices, extrema, label=f"Simulated cumulated max {quantity_name}")
    plt.plot(t, quantities, label=f"Simulated {quantity_name}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(f"Amplitude of {quantity_name}")
    plt.show()


def visualize_proc_vs_MLUPS(save: bool = False, format: str = 'pdf') -> None:
    plt.figure(figsize=(15, 5))

    for size, col in zip([100, 300, 1000], ['blue', 'red', 'black']):
        dir_name = "log/sliding_lid_W0.10_visc0.03_size{0}x{0}_parallel/".format(size)
        file_name = 'MLUPS_vs_proc.csv'

        file_path = f"{dir_name}{file_name}"

        with open(file_path, 'r') as f:
            reader = list(csv.reader(f, delimiter=','))
            procs = np.array([int(row[0]) for row in reader])
            mlups = np.array([float(row[1]) / 1e6 for row in reader])

        order = np.argsort(procs)
        procs, mlups = procs[order], mlups[order]
        plt.plot(procs, mlups, marker='x', color=col, label='$X \\times Y = {0} \\times {0}$'.format(size))

    plt.xlabel('# of processes', fontsize=28)
    plt.ylabel('MLUPS', fontsize=28)
    plt.ylim(1., 1000)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend()
    plt.grid()
    show_or_save(path=f'log/scaling_test.{format}' if save else None)
