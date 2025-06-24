"""
This module provides functionality for managing microphone calibration data.

It enables amplitude correction of recorded signals by applying
frequency-dependent scaling and phase-correction based on a stored calibration
curve, supporting accurate pressure-level measurements in OAE recordings.

Note:
    This is a dummy module. Functionality will be added in future revisions.‚
"""


from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

@dataclass
class MicroTransferFunction:
    """Store the transfer function of the microphone."""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the transfer function in Hz."""

    amplitudes: npt.NDArray[np.float32]
    """Amplitudes of the transfer function in full-scale/muPa"""

    phases: npt.NDArray[np.float32]|None=None
    """Phases of the transfer function in radiant."""
