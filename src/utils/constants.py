from typing import Callable, Tuple

import numpy as np


EquationFuncType = Callable[[Tuple[np.ndarray, np.ndarray]], np.ndarray]
