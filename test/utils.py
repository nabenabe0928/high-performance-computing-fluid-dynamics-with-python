import numpy as np


def abssum(a1: np.ndarray, a2: np.ndarray) -> float:
    """ return the sum of the absolute different of each element """
    assert a1.shape == a2.shape
    return np.abs(a1 - a2).sum()
