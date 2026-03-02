"""Module with math functions for PyOAE DSP."""

import numpy as np
import numpy.typing as npt


def rms(y: npt.NDArray[np.float32 | np.float64]) -> float:
    """Calculates the RMS for a given signal."""
    return np.sqrt(np.mean(np.square(y)))
