"""
This module provides functionality for managing microphone calibration data.

It enables amplitude correction of recorded signals by applying
frequency-dependent scaling and phase-correction based on a stored calibration
curve, supporting accurate pressure-level measurements in OAE recordings.

Note:
    This is a dummy module. Functionality will be added in future revisions.‚
"""


from dataclasses import dataclass
#from typing import TypedDict

import numpy as np
import numpy.typing as npt

# from pyoae.device.device_config import DeviceConfig


@dataclass
class OutputCalibration:
    """Linear scaling functions to apply output calibration."""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the output sensitivity function."""

    sensitivity: npt.NDArray[np.float32]
    """Output sensitivity function (transfer function).

    This is a 2D array of dimensions [num_ch, num_bins]
    """

    def get_sensitivity(self, ch: int, f: float) -> float:
        """Returns the output sensitivity in DFS/muPa.

        Args:
            ch: index of the output channel starting at 0
            f: frequency of the output stimulus
        """
        if ch >= self.sensitivity.shape[0]:
            # TODO: log error
            return 0.0

        # TODO: check frequency boundaries
        # find frequency-bin index
        # (alternatively, we could store the frequency resolution
        # in order to calculate the frequency-bin index)
        idx = np.argmin(np.abs(self.frequencies - f))
        return self.sensitivity[ch, idx]

    def pressure_to_full_scale(self, ch: int, p: float, f: float) -> float:
        """Calculates digital full-scale amplitude from peak pressure."""
        s = self.get_sensitivity(ch, f)
        return p * s


@dataclass
class MicroTransferFunction:
    """Store the transfer function of the microphone."""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the transfer function in Hz."""

    amplitudes: npt.NDArray[np.float32]
    """Amplitudes of the transfer function in full-scale/muPa"""

    phases: npt.NDArray[np.float32]|None=None
    """Phases of the transfer function in radiant."""
