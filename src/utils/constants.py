from typing import Callable, Tuple
from enum import IntEnum

import numpy as np

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod


EquationFuncType = Callable[[Tuple[np.ndarray, np.ndarray]], np.ndarray]
BoundaryHandlingFuncType = Callable[[LatticeBoltzmannMethod], None]


class DirectionIndicators(IntEnum):
    RIGHT: int = 0
    LEFT: int = 1
    TOP: int = 2
    BOTTOM: int = 3
