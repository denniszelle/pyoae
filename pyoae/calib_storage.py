"""Module providing functionality for managing calibration data.

It enables amplitude correction of recorded signals by applying
frequency-dependent scaling and phase-correction based on a stored calibration
curve, supporting accurate pressure-level measurements in OAE recordings.

Note:
    Phase calibration will be added in future revisions of PyOAE.
"""

from typing import TypedDict, TypeVar
from logging import Logger

import numpy as np
import numpy.typing as npt
from scipy import interpolate

from pyoae import get_logger
from pyoae.device.device_config import DeviceConfig

TFloat = TypeVar('TFloat', bound=np.floating)


def interpolate_tf(
    interp_freqs: npt.NDArray[TFloat],
    raw_freqs: npt.NDArray[TFloat],
    raw_values: npt.NDArray[TFloat]
) -> npt.NDArray[TFloat]:
    """Interpolate a transfer function onto new frequency values

    Args:
        interp_freqs: Target frequencies where the transfer function should be
            evaluated. Must be a 1-D array of NumPy floating dtype.
        raw_freqs: Frequencies of the known transfer function samples.
            Must be a 1-D, strictly increasing array with the same dtype as
            ``interp_freqs``.
        raw_values: Transfer function values corresponding to ``raw_freqs``.
            Must have the same shape and dtype as ``raw_freqs``.

    Returns:
        Interpolated transfer function evaluated at ``interp_freqs``.
        The returned array has the same shape and floating dtype as
        ``interp_freqs``.

    """

    amp_spline = interpolate.CubicSpline(
        raw_freqs,
        raw_values,
        bc_type='natural',
        extrapolate=False
    )
    values_ip = amp_spline(interp_freqs)
    values_ip[interp_freqs < raw_freqs[0]] = raw_values[0]
    values_ip[interp_freqs > raw_freqs[-1]] = raw_values[-1]

    # values_ip = np.interp(interp_freqs, raw_freqs, raw_values)

    return values_ip


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


class BaseTransferFunction:
    """Base class for frequency-dependent complex transfer functions.

    This class provides shared functionality for interpolating amplitude and
    phase responses onto arbitrary frequency grids and constructing complex
    transfer functions.

    Subclasses are responsible for loading calibration data and exposing
    user-facing APIs (e.g., channel handling or pressure conversions).
    """

    logger: Logger
    """Logger used for debug, warning and error messages."""

    raw_freqs: npt.NDArray[np.float32]
    """Frequencies of the measured transfer function in Hz."""

    raw_amps: npt.NDArray[np.float32]
    """Measured amplitudes of the transfer function."""

    raw_phases: npt.NDArray[np.float32]
    """Measured phase of the transfer function in radians."""

    def __init__(self, log: Logger | None = None) -> None:
        self.logger = log or get_logger()

    # ------------------------------------------------------------------
    # Helper methods used by subclasses
    # ------------------------------------------------------------------

    def _get_freq_grid(
        self,
        frequencies_ip: npt.NDArray[np.float32] | None,
        num_samples: int | None
    ) -> npt.NDArray[np.float32]:
        """Return a frequency grid for interpolation.

        If a grid is provided it is returned unchanged. Otherwise a grid is
        generated using ``np.fft.rfftfreq``.

        Args:
            frequencies_ip: Optional frequency grid in Hz.
            num_samples: Number of time-domain samples used to generate the FFT
                frequency grid when ``frequencies_ip`` is ``None``.

        Returns:
            Frequency grid in Hz. Returns an empty array if neither argument
            is provided and logs an error.
        """
        if frequencies_ip is not None:
            return frequencies_ip

        if num_samples is None:
            self.logger.error(
                "Neither frequencies nor number of samples given."
            )
            return np.ndarray(0, np.float32)

        return np.fft.rfftfreq(num_samples, 1 / DeviceConfig.sample_rate)

    def _interp_amp_phase(
        self,
        freqs: npt.NDArray[np.float32],
        amps: npt.NDArray[np.float32],
        phases: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.complex64]:
        """Interpolate amplitude and phase and build a complex TF.

        Args:
            freqs: Frequencies where the transfer function should be evaluated.
            amps: Amplitude response corresponding to ``self.raw_freqs``.
            phases: Phase response corresponding to ``self.raw_freqs``.

        Returns:
            Complex transfer function evaluated at ``freqs``.
        """
        amp_ip = interpolate_tf(freqs, self.raw_freqs, amps)
        phase_ip = interpolate_tf(freqs, self.raw_freqs, phases)

        return (
            np.array(amp_ip, dtype=np.complex64)
            * np.exp(1j * phase_ip, dtype=np.complex64)
        )


class MicroTransferFunction(BaseTransferFunction):
    """Interpolated microphone transfer function."""

    def __init__(
        self,
        abs_calib: AbsCalibData,
        trans_fun: MicroTransferFunData,
        log: Logger | None = None
    ) -> None:
        super().__init__(log)

        self.raw_freqs = np.array(trans_fun['frequencies'], np.float32)
        self.raw_amps = np.array(trans_fun['amplitudes'], np.float32)
        self.raw_phases = np.array(trans_fun['phases'], np.float32)

        # Convert to DFS / µPa
        self.raw_amps /= abs_calib['sensitivity']

    def get_interp_transfer_function(
        self,
        frequencies_ip: npt.NDArray[np.float32] | None = None,
        num_samples: int | None = None
    ) -> npt.NDArray[np.complex64]:
        """Return the interpolated microphone transfer function.

        Args:
            frequencies_ip: Frequencies in Hz where the transfer function should
                be evaluated.
            num_samples: Number of samples used to generate an FFT frequency
                grid when ``frequencies_ip`` is ``None``.

        Returns:
            Complex microphone transfer function.
        """

        freqs = self._get_freq_grid(frequencies_ip, num_samples)
        return self._interp_amp_phase(freqs, self.raw_amps, self.raw_phases)

    def get_sensitivity(self, f: float) -> float:
        """Return microphone sensitivity at a given frequency.

        Args:
            f: Frequency in Hz.

        Returns:
            Sensitivity in DFS/µPa.
        """

        amp = self.get_interp_transfer_function(np.asarray([f], np.float32))
        return float(np.abs(amp[0]))


class OutputCalibration(BaseTransferFunction):
    """Interpolated loudspeaker/output transfer function."""

    output_channels: list[int]
    """Output Channels the output calibration was performed on"""

    input_channels: list[int]
    """Input Channels the output calibration was performed on"""

    date: str
    """Time stamp of output calibration."""

    def __init__(
        self,
        calib_data: SpeakerCalibData,
        log: Logger | None = None
    ) -> None:
        super().__init__(log)

        self.date = calib_data['date']

        self.raw_freqs = np.array(calib_data['frequencies'], np.float32)
        self.raw_amps = np.array(calib_data['max_out'], np.float32)
        self.raw_phases = np.array(calib_data['phase'], np.float32)

        self.output_channels = calib_data['output_channels']
        self.input_channels = calib_data['input_channels']

    def get_interp_transfer_function(
        self,
        channel: int,
        frequencies_ip: npt.NDArray[np.float32] | None = None,
        num_samples: int | None = None,
    ) -> npt.NDArray[np.complex64]:
        """Return interpolated transfer function for an output channel.

        Args:
            channel: Output channel index.
            frequencies_ip: Frequencies in Hz where the transfer function
                should be evaluated.
            num_samples: Number of samples used to generate an FFT frequency
                grid when ``frequencies_ip`` is ``None``.

        Returns:
            Complex transfer function of the selected channel.
        """
        idx = self.output_channels.index(channel)
        freqs = self._get_freq_grid(frequencies_ip, num_samples)

        return self._interp_amp_phase(
            freqs,
            self.raw_amps[idx],
            self.raw_phases[idx],
        )

    def get_sensitivity(self, ch: int, f: float) -> float:
        """Return the calibrated output sensitivity for a channel at a frequency.

        Args:
            ch: Output channel index (starting at 0).
            f: Frequency of the output stimulus in Hz.

        Returns:
            Sensitivity in DFS/µPa for the requested channel and frequency.
            Returns 0.0 if the channel is not calibrated.
        """
        if ch not in self.output_channels:
            self.logger.error("Output channel %s was not calibrated.", ch)
            return 0.0

        if f < np.min(self.raw_freqs) or f > np.max(self.raw_freqs):
            self.logger.warning(
                'Stimulus frequency %.2f Hz outside calibrated boundaries.',
                f
            )

        amp = self.get_interp_transfer_function(
            ch,
            np.asarray([f], np.float32)
        )
        return float(np.abs(amp[0]))

    def pressure_to_full_scale(self, ch: int, p: float, f: float) -> float:
        """Convert peak acoustic pressure to digital full-scale amplitude (FS).

        Args:
            ch: Output channel index (starting at 0).
            p: Peak acoustic pressure in µPa.
            f: Frequency of the stimulus in Hz.

        Returns:
            Digital full-scale amplitude required to produce the given
            pressure on the selected output channel.
        """
        s = abs(self.get_sensitivity(ch, f))
        return p / s
