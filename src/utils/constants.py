from typing import Any, Callable, Dict, List, Tuple
from enum import IntEnum

import numpy as np


EquationFuncType = Callable[[np.ndarray], np.ndarray]


def viscosity_equation(t: int, epsilon: float, velocity: np.ndarray) -> np.ndarray:
    """
    v(y, t) = epsilon * exp(- visc * (2pi / Y) ** 2 * t) sin(2pi / Y * y)
    => - visc * (2pi / Y) ** 2 * t = log(v(y, t) / sin(2pi / Y * y) / epsilon)
    """
    assert len(velocity.shape) == 1
    Y = velocity.shape[0]
    y = np.arange(Y)
    coef = 2 * np.pi / Y
    visc = - np.log(velocity / np.sin(coef * y) / epsilon) / coef ** 2 / t
    return visc


def density_equation(epsilon: float, viscosity: float, lattice_grid_shape: Tuple[int, int]
                     ) -> EquationFuncType:
    X, _ = lattice_grid_shape

    def _imp(t: np.ndarray) -> np.ndarray:
        return epsilon * np.exp(-viscosity * (2 * np.pi / X) ** 2 * t)

    return _imp


def velocity_equation(epsilon: float, viscosity: float, lattice_grid_shape: Tuple[int, int]
                      ) -> EquationFuncType:
    _, Y = lattice_grid_shape

    def _imp(t: np.ndarray) -> np.ndarray:
        return epsilon * np.exp(-viscosity * (2 * np.pi / Y) ** 2 * t)

    return _imp


def sinusoidal_velocity(lattice_grid_shape: Tuple[int, int], epsilon: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return initial values according to
    rho(x, y, 0) = 1 and v(x, y, 0) = eps * sin(2PI * y/Y)

    Args:
        epsilon: amplitude of swinging

    Returns:
        density := rho(x, y, 0) = 1
        v(x, y, 0) = eps * sin(2PI * y/Y)

    Note:
        constraint |v| < 0.1
    """
    assert abs(epsilon) < 0.1

    X, Y = lattice_grid_shape
    density = np.ones(lattice_grid_shape)
    vel = np.zeros((*lattice_grid_shape, 2))
    vx = epsilon * np.sin(2 * np.pi * np.arange(Y) / Y)
    vel[..., 0] = np.tile(vx, (X, 1))

    return density, vel


def sinusoidal_density(lattice_grid_shape: Tuple[int, int], epsilon: float,
                       rho0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return initial values according to
    rho(x, y, 0) = rho0 + eps*sin(2PI x/X)
    and v(x, y, 0) = 0

    Args:
        epsilon: amplitude of swinging
        rho0: The offset of density

    Returns:
        density := rho(x, y, 0) = rho0 + eps * sin(2PI x/X)
        v(x, y, 0) = 0
    """
    assert rho0 + epsilon < 1
    assert rho0 - epsilon > 0
    X, Y = lattice_grid_shape
    vel = np.zeros((*lattice_grid_shape, 2))
    density_x = rho0 + epsilon * np.sin(2 * np.pi * np.arange(X) / X)
    density = np.tile(density_x, (Y, 1)).T

    return density, vel


class DirectionIndicators(IntEnum):
    CENTER: int = 0
    RIGHT: int = 1
    TOP: int = 2
    LEFT: int = 3
    BOTTOM: int = 4
    RIGHTTOP: int = 5
    LEFTTOP: int = 6
    LEFTBOTTOM: int = 7
    RIGHTBOTTOM: int = 8

    def is_left(self) -> bool:
        return 'LEFT' in self.name

    def is_right(self) -> bool:
        return 'RIGHT' in self.name

    def is_top(self) -> bool:
        return 'TOP' in self.name

    def is_bottom(self) -> bool:
        return 'BOTTOM' in self.name

    def is_center(self, axis: int) -> bool:
        """
        axis == 0: x axis
        axis == 1: y axis
        """
        if axis == 0:
            return self in [
                DirectionIndicators.CENTER,
                DirectionIndicators.RIGHT,
                DirectionIndicators.LEFT
            ]
        else:
            return self in [
                DirectionIndicators.CENTER,
                DirectionIndicators.TOP,
                DirectionIndicators.BOTTOM
            ]

    def is_opposite(self, dir: 'DirectionIndicators') -> bool:
        """ dir points the opposite side of self. """
        if self.is_right() and dir.is_left():
            return True
        elif self.is_left() and dir.is_right():
            return True
        elif self.is_bottom() and dir.is_top():
            return True
        elif self.is_top() and dir.is_bottom():
            return True
        else:
            return False

    def is_sameside(self, dir: 'DirectionIndicators') -> bool:
        if self.is_right() and dir.is_right():
            return True
        elif self.is_left() and dir.is_left():
            return True
        elif self.is_bottom() and dir.is_bottom():
            return True
        elif self.is_top() and dir.is_top():
            return True
        else:
            return False


class MetaAdjacentAttributes(type):
    """
    The attributes for the adjacent cells.
    such as the following indices are:
    y_upper  -> 6 2 5
    y_center -> 3 0 1
    y_lower  -> 7 4 8
    """
    _x_left = np.array([dir for dir in DirectionIndicators if dir.is_left()])
    _x_right = np.array([dir for dir in DirectionIndicators if dir.is_right()])
    _y_top = np.array([dir for dir in DirectionIndicators if dir.is_top()])
    _y_bottom = np.array([dir for dir in DirectionIndicators if dir.is_bottom()])

    _x_center = np.array([
        DirectionIndicators.CENTER,
        DirectionIndicators.TOP,
        DirectionIndicators.BOTTOM
    ])
    _y_center = np.array([
        DirectionIndicators.CENTER,
        DirectionIndicators.RIGHT,
        DirectionIndicators.LEFT
    ])

    _reflected_direction = np.array([
        DirectionIndicators.CENTER,
        DirectionIndicators.LEFT,
        DirectionIndicators.BOTTOM,
        DirectionIndicators.RIGHT,
        DirectionIndicators.TOP,
        DirectionIndicators.LEFTBOTTOM,
        DirectionIndicators.RIGHTBOTTOM,
        DirectionIndicators.RIGHTTOP,
        DirectionIndicators.LEFTTOP
    ])
    _velocity_direction_set = np.array([
        [0, 0], [1, 0], [0, 1],
        [-1, 0], [0, -1], [1, 1],
        [-1, 1], [-1, -1], [1, -1]
    ])

    _weights = np.array(
        [4. / 9.]
        + [1. / 9.] * 4
        + [1. / 36.] * 4
    )

    def __init__(cls, *args: List[Any], **kwargs: Dict[str, Any]):
        pass

    @property
    def x_left(cls) -> np.ndarray:
        return cls._x_left

    @property
    def x_center(cls) -> np.ndarray:
        return cls._x_center

    @property
    def x_right(cls) -> np.ndarray:
        return cls._x_right

    @property
    def y_top(cls) -> np.ndarray:
        return cls._y_top

    @property
    def y_center(cls) -> np.ndarray:
        return cls._y_center

    @property
    def y_bottom(cls) -> np.ndarray:
        return cls._y_bottom

    @property
    def velocity_direction_set(cls) -> np.ndarray:
        """ Note: Those do not have identical norms. """
        return cls._velocity_direction_set

    @property
    def reflected_direction(cls) -> np.ndarray:
        return cls._reflected_direction

    @property
    def weights(cls) -> np.ndarray:
        """ The weights for each adjacent cell """
        return cls._weights


class AdjacentAttributes(metaclass=MetaAdjacentAttributes):
    """ From this class, you can call properties above """
    pass
