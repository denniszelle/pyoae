"""Module providing functionality for managing calibration data.

It enables amplitude correction of recorded signals by applying
frequency-dependent scaling and phase-correction based on a stored calibration
curve, supporting accurate pressure-level measurements in OAE recordings.

Note:
    Phase calibration will be added in future revisions of PyOAE.
"""

from typing import TypedDict
from logging import Logger
from numbers import Complex

import numpy as np
import numpy.typing as npt

from pyoae import get_logger
from pyoae.device.device_config import DeviceConfig

class AbsCalibData(TypedDict):
    """Container for absolute-calibration data in calibration file."""

    date: str
    """Date of the absolute calibration"""

    ref_frequency: float
    """Reference frequency of the absolute calibration"""

    sensitivity: float
    """Sensitivity of the microphone at the reference frequency"""

    calib_type: int
    """Identifier of the calibration type."""


class MicroTransferFunData(TypedDict):
    """Container for transfer-function data in calibration file."""

    date: str
    """Date of the microphone transfer function calibration"""

    frequencies: list[float]
    """Frequencies for which the transfer function has been evaluated"""

    amplitudes: list[float]
    """Amplitudes of the transfer function"""

    phases: list[float]
    """Phase values of the transfer function"""


class MicroCalibData(TypedDict):
    """Container to load a microphone calibration file."""

    doc_type: str
    """Document type the loaded calibration"""

    rev: int
    """Revision number of the document"""

    probe_sn: str
    """Serial number of the probe"""

    model: str
    """Model identifier of the probe"""

    side: str
    """Measurement side for the probe"""

    abs_calibration: AbsCalibData
    """Sensitivity calibration of the probe"""

    transfer_function: MicroTransferFunData
    """Transfer function calibration of the probe"""


class SpeakerCalibData(TypedDict):
    """Container to save/load speaker calibration data.

    Container is compatible with JSON export.
    """
    date: str
    """Date the output calibration was performed"""

    output_channels: list[int]
    """Channels the output calibration was performed on"""

    input_channels: list[int]
    """Channels the output calibration was performed on"""

    frequencies: list[float]
    """Measured frequencies of the multitones"""

    max_out: list[list[float]]
    """Maximum amplitude output"""

    phase: list[list[float]]
    """Phase shift of the first given channel"""


def get_empty_micro_calib_data() -> MicroCalibData:
    """Returns an empty container for microphone-calibration data."""
    a: AbsCalibData = {
        'date': '',
        'ref_frequency': 1000.0,
        'sensitivity': 1,
        'calib_type': 2
    }
    t: MicroTransferFunData = {
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


def get_empty_speaker_calib_data() -> SpeakerCalibData:
    """Returns an empty container for speaker-calibration data."""
    d: SpeakerCalibData = {
        'date': '',
        'output_channels': [],
        'input_channels': [],
        'frequencies': [],
        'max_out': [],
        'phase': []
    }
    return d


class OutputCalibration:
    """Linear scaling functions to apply output calibration."""

    logger: Logger
    """Class logger for debug, info, warning and error messages"""

    raw_freqs: npt.NDArray[np.float32]
    """Frequencies of the output sensitivity function."""

    raw_amps: npt.NDArray[np.float32]
    """Output sensitivity function (transfer function).

    This is a 2D array of dimensions [num_ch, num_bins]
    """

    raw_phases: npt.NDArray[np.float32]
    """Phases of the transfer function in radiant."""

    output_channels: list[int]
    """Output channels used for calibration"""

    input_channels: list[int]
    """Input channels used for calibration"""

    date: str
    """Time stamp of output calibration."""

    def __init__(
        self,
        calib_data: SpeakerCalibData,
        log: Logger | None = None
    ) -> None:
        """Initializes an scaled input-channel transfer function."""

        self.logger = log or get_logger()

        self.date = calib_data['date']
        self.raw_freqs = np.array(
            calib_data['frequencies'], dtype=np.float32
        )
        self.raw_amps = np.array(calib_data['max_out'])
        self.raw_phases = np.array(calib_data['phase'], dtype=np.float32)

        self.output_channels = calib_data['output_channels']
        self.input_channels = calib_data['input_channels']

    def get_sensitivity(self, ch: int, f: float) -> float:
        """Returns the output sensitivity in DFS/muPa.

        Args:
            ch: index of the output channel starting at 0
            f: frequency of the output stimulus
        """

        if ch not in self.output_channels:
            self.logger.error('Output channel was not calibrated.')
            return 0.0

        ch_idx = self.output_channels.index(ch)

        if f < np.min(self.raw_freqs) or f > np.max(self.raw_freqs):
            self.logger.warning(
                'Stimulus frequency %s Hz outside calibrated boundaries.',
                f
            )

        # find frequency-bin index
        # (alternatively, we could store the frequency resolution
        # in order to calculate the frequency-bin index)
        idx = np.argmin(np.abs(self.raw_freqs - f))
        return self.raw_amps[ch_idx, idx]

    def pressure_to_full_scale(self, ch: int, p: float, f: float) -> float:
        """Calculates digital full-scale amplitude from peak pressure."""
        s = abs(self.get_sensitivity(ch, f))
        return p / s

    def get_interp_transfer_function(
        self,
        channel: int,
        frequencies_ip: npt.NDArray[np.float32] | None = None,
        num_samples: int | None = None
    ) -> npt.NDArray[np.complex64]:
        """Return interpolated transfer function"""

        index = self.output_channels.index(channel)

        if frequencies_ip is None:
            if num_samples is None:
                self.logger.error(
                    'Neither frequencies nor number of samples given.'
                )
                return np.ndarray(0, np.complex64)
            frequencies_ip = np.fft.rfftfreq(
                num_samples, 1/DeviceConfig.sample_rate
            )

        amplitudes_ip = np.interp(
            frequencies_ip, self.raw_freqs, self.raw_amps[index]
        )
        phases_ip = np.interp(frequencies_ip, self.raw_freqs, self.raw_phases[index])

        interpolated_tf = (
            np.array(amplitudes_ip, dtype=np.complex64)
            *np.exp(1j * phases_ip, dtype=np.complex64)
        )
        return interpolated_tf


class MicroTransferFunction:
    """Interpolated transfer function of the microphone."""

    logger: Logger
    """Class logger for debug, info, warning and error messages"""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the transfer function in Hz."""

    amplitudes: npt.NDArray[np.complex64]
    """Amplitudes of the transfer function in full-scale/muPa"""

    phases: npt.NDArray[np.float32]
    """Phases of the transfer function in radiant."""

    def __init__(
        self,
        abs_calib: AbsCalibData,
        trans_fun: MicroTransferFunData,
        log: Logger | None = None
    ) -> None:
        """Initializes an scaled input-channel transfer function."""

        self.logger = log or get_logger()

        self.raw_freqs = np.array(trans_fun['frequencies'], dtype=np.float32)
        self.raw_amps = np.array(trans_fun['amplitudes'], dtype=np.float32)
        self.raw_phases = np.array(trans_fun['phases'], dtype=np.float32)

        if np.max(np.abs(np.diff(self.raw_phases))) > np.pi:
            self.logger.error(
                'Phase difference between two points larger than pi. '
                'This may result in unintended wrapping.'
            )

        # scale amplitudes to DFS/muPa
        self.raw_amps /= abs_calib['sensitivity']


    def get_sensitivity(self, f: float) -> Complex:
        """Returns the output sensitivity in DFS/muPa.

        Args:
            f: frequency at which transfer function should be sampled
        """

        if f < np.min(self.frequencies) or f > np.max(self.frequencies):
            self.logger.warning(
                '%s Hz outside microphone calibrated boundaries.',
                f
            )

        # find frequency-bin index
        # (alternatively, we could store the frequency resolution
        # in order to calculate the frequency-bin index)
        idx = np.argmin(np.abs(self.frequencies - f))
        return self.amplitudes[idx]

    def get_interp_transfer_function(
        self,
        frequencies_ip: npt.NDArray[np.float32] | None = None,
        num_samples: int | None = None
    ) -> npt.NDArray[np.complex64]:
        """Return interpolated transfer function"""
        if frequencies_ip is None:
            if num_samples is None:
                self.logger.error(
                    'Neither frequencies nor number of samples given.'
                )
                return np.ndarray(0, np.complex64)
            frequencies_ip = np.fft.rfftfreq(
                num_samples, 1/DeviceConfig.sample_rate
            )

        amplitudes_ip = np.interp(
            frequencies_ip, self.raw_freqs, self.raw_amps
        )
        phases_ip = np.interp(frequencies_ip, self.raw_freqs, self.raw_phases)

        interpolated_tf = (
            np.array(amplitudes_ip, dtype=np.complex64)
            *np.exp(1j * phases_ip, dtype=np.complex64)
        )
        return interpolated_tf
