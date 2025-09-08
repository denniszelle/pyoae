"""
This module provides functionality for managing microphone calibration data.

It enables amplitude correction of recorded signals by applying
frequency-dependent scaling and phase-correction based on a stored calibration
curve, supporting accurate pressure-level measurements in OAE recordings.

Note:
    This is a dummy module. Functionality will be added in future revisions.‚
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

# from pyoae.device.device_config import DeviceConfig


class AbsCalibData(TypedDict):
    """Container for absolute-calibration data in calibration file."""
    date: str
    ref_frequency: float
    sensitivity: float
    calib_type: int


class TransferFunData(TypedDict):
    """Container for transfer-function data in calibration file."""
    date: str
    frequencies: list[float]
    amplitudes: list[float]
    phases: list[float]


class MicroCalibData(TypedDict):
    """Container to load a microphone calibration file."""
    doc_type: str
    rev: int
    probe_sn: str
    model: str
    side: str
    abs_calibration: AbsCalibData
    transfer_function: TransferFunData


def get_empty_micro_calib_data() -> MicroCalibData:
    """Returns an empty container for microphone-calibration data."""
    a: AbsCalibData = {
        'date': '',
        'ref_frequency': 1000.0,
        'sensitivity': 1,
        'calib_type': 2
    }
    t: TransferFunData = {
        'date': '',
        'frequencies': [1.0, 20000.0],
        'amplitudes': [1.0, 1.0],
        'phases': [0.0, 0.0]
    }
    d: MicroCalibData = {
        'doc_type': '',
        'rev': 2,
        'probe_sn': '',
        'model': '',
        'side': '',
        'abs_calibration': a,
        'transfer_function': t
    }
    return d


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


class MicroTransferFunction:
    """Interpolated transfer function of the microphone."""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the transfer function in Hz."""

    amplitudes: npt.NDArray[np.float32]
    """Amplitudes of the transfer function in full-scale/muPa"""

    phases: npt.NDArray[np.float32]
    """Phases of the transfer function in radiant."""

    num_samples: int | None
    """Number of samples for which the transfer function was interpolated."""

    sample_rate: float | None
    """Sample rate in Hz for which the transfer function was interpolated"""

    def __init__(
        self,
        abs_calib: AbsCalibData,
        trans_fun: TransferFunData,
        num_samples: int | None = None,
        sample_rate: float | None = None
    ) -> None:
        """Initializes an scaled input-channel transfer function."""
        self.frequencies = np.array(trans_fun['frequencies'], dtype=np.float32)
        self.amplitudes = np.array(trans_fun['amplitudes'], dtype=np.float32)
        self.phases = np.array(trans_fun['phases'], dtype=np.float32)

        # scale amplitudes to DFS/muPa
        self.amplitudes /= abs_calib['sensitivity']

        self.num_samples = num_samples
        self.sample_rate = sample_rate

    def interpolate_transfer_fun(self) -> None:
        """Interpolates the transfer function """
        if self.num_samples is None or self.sample_rate is None:
            return

        num_bins = (self.num_samples // 2) + 1
        df = self.sample_rate / self.num_samples
        frequencies_ip = np.arange(num_bins, dtype=np.float32) * df

        amplitudes_ip = np.interp(frequencies_ip, self.frequencies, self.amplitudes)
        phases_ip = np.interp(frequencies_ip, self.frequencies, self.phases)

        # store interpolated data
        self.frequencies = frequencies_ip
        self.amplitudes = amplitudes_ip.astype(np.float32)
        self.phases = phases_ip.astype(np.float32)
