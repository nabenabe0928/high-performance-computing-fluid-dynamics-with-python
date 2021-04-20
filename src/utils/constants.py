from typing import Callable, Tuple

import numpy as np

from src.simulation_attributes.lattice_boltzmann_method import LatticeBoltzmannMethod


EquationFuncType = Callable[[Tuple[np.ndarray, np.ndarray]], np.ndarray]
BoundaryHandlingFuncType = Callable[[LatticeBoltzmannMethod], None]
