"""Functions and classes to generate and manage output signals."""

import numpy as np
# import numpy.typing as npt

# TODO: add overwrites for typing of arrays

def db_to_lin(level: float) -> float:
    """Convert logarithmic value from dB to linear.

    This function can be used to scale from dBFS to peak amplitude.
    """
    return 10**(level/20)


def db_spl_to_peak_mupa(level: float) -> float:
    """Converts sound pressure level to peak signal amplitude in muPa."""
    return 20 * np.sqrt(2) * 10**(level/20)


def db_spl_to_rms_mupa(level: float) -> float:
    """Converts sound pressure level to signal RMS in muPa."""
    return 20 * 10**(level/20)
