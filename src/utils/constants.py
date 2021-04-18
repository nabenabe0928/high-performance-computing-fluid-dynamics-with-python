from typing import Callable, Tuple

import numpy as np

from src.simulation_attributes.formula import FluidField2D


EquationFuncType = Callable[[Tuple[np.ndarray, np.ndarray]], np.ndarray]
BoundaryHandlingFuncType = Callable[[FluidField2D], None]
