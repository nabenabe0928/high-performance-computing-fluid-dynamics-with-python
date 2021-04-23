from typing import Any, Callable, Dict, List, Tuple
from enum import IntEnum

import numpy as np


EquationFuncType = Callable[[Tuple[np.ndarray, np.ndarray]], np.ndarray]


class DirectionIndicators(IntEnum):
    RIGHT: int = 0
    LEFT: int = 1
    TOP: int = 2
    BOTTOM: int = 3
    RIGHTTOP: int = 4
    LEFTTOP: int = 5
    RIGHTBOTTOM: int = 6
    LEFTBOTTOM: int = 7


DIRECTION2VEC = np.array([
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, 1], [1, -1], [-1, -1]
    ], dtype=np.int32)


class MetaAdjacentAttributes(type):
    """
    The attributes for the adjacent cells.
    such as the following indices are:
    y_upper  -> 6 2 5
    y_center -> 3 0 1
    y_lower  -> 7 4 8
    """
    def __init__(cls, *args: List[Any], **kwargs: Dict[str, Any]):
        pass

    @property
    def x_left(cls) -> np.ndarray:
        return np.array([3, 6, 7])

    @property
    def x_center(cls) -> np.ndarray:
        return np.array([0, 2, 4])

    @property
    def x_right(cls) -> np.ndarray:
        return np.array([1, 5, 8])

    @property
    def y_upper(cls) -> np.ndarray:
        return np.array([2, 5, 6])

    @property
    def y_center(cls) -> np.ndarray:
        return np.array([0, 1, 3])

    @property
    def y_lower(cls) -> np.ndarray:
        return np.array([4, 7, 8])

    @property
    def velocity_direction_set(cls) -> np.ndarray:
        """ Note: Those do not have identical norms. """
        return np.array([[0, 0], [1, 0], [0, 1],
                         [-1, 0], [0, -1], [1, 1],
                         [-1, 1], [-1, -1], [1, -1]])

    @property
    def reflected_direction(cls) -> np.ndarray:
        return np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    @property
    def weights(cls) -> np.ndarray:
        """ The weights for each adjacent cell """
        return np.array(
            [4. / 9.]
            + [1. / 9.] * 4
            + [1. / 36.] * 4
        )


class AdjacentAttributes(metaclass=MetaAdjacentAttributes):
    """ From this class, you can call properties above """
    pass
