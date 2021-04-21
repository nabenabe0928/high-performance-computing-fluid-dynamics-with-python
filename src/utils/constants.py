from typing import Callable, Tuple
from enum import IntEnum

import numpy as np


EquationFuncType = Callable[[Tuple[np.ndarray, np.ndarray]], np.ndarray]


class DirectionIndicators(IntEnum):
    RIGHT: int = 0
    LEFT: int = 1
    TOP: int = 2
    BOTTOM: int = 3


DIRECTION2VEC = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)
