"""Module providing functionality for managing calibration data during runtime.

It provides an interpolated transferfunction for speaker and microphone
corrections.

"""

from abc import ABC
from logging import Logger

import numpy as np
import numpy.typing as npt
from scipy import interpolate

from pyoae import get_logger
from pyoae.calib_storage import (
    AbsCalibData,
    MicroTransferFunData,
    SpeakerCalibData,
)
from pyoae.device.device_config import DeviceConfig


class BaseTransferFunction(ABC):
    """Base class for frequency-dependent complex transfer functions.

    This class provides shared functionality for interpolating amplitude and
    phase responses onto arbitrary frequency grids and constructing complex
    transfer functions.

    Subclasses are responsible for loading calibration data and exposing
    user-facing APIs (e.g., channel handling or pressure conversions).
    """

    logger: Logger
    """Logger used for debug, warning and error messages."""

    def __init__(self, log: Logger | None = None) -> None:
        self.logger = log or get_logger()

    # ------------------------------------------------------------------
    # Helper methods used by subclasses
    # ------------------------------------------------------------------

    def _get_freq_grid(
        self,
        frequencies_ip: npt.NDArray[np.float32] | None,
        num_samples: int | None,
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
        intep_freqs: npt.NDArray[np.float32],
        raw_freqs: npt.NDArray[np.float32],
        raw_amps: npt.NDArray[np.float32],
        raw_phases: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.complex64]:
        """Interpolate amplitude and phase and build a complex TF.

        Args:
            intep_freqs: Frequencies where the transfer function should be evaluated.
            raw_freqs: Frequencies of the given data
            raw_amps: Amplitudes of the given data corresponding to raw_freqs.
            raw_phases: Phases of the given data corresponding to raw_freqs.

        Returns:
            Complex transfer function evaluated at ``freqs``.
        """
        amp_ip = self.interpolate_tf(intep_freqs, raw_freqs, raw_amps)
        phase_ip = self.interpolate_tf(intep_freqs, raw_freqs, raw_phases)

        return np.array(amp_ip, dtype=np.complex64) * np.exp(
            1j * phase_ip, dtype=np.complex64
        )

    def interpolate_tf(
        self,
        interp_freqs: npt.NDArray[np.float32],
        raw_freqs: npt.NDArray[np.float32],
        raw_values: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
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
            raw_freqs, raw_values, bc_type='natural', extrapolate=False
        )
        values_ip = amp_spline(interp_freqs)
        values_ip[interp_freqs < raw_freqs[0]] = raw_values[0]
        values_ip[interp_freqs > raw_freqs[-1]] = raw_values[-1]

        return values_ip


class MicroTransferFunction(BaseTransferFunction):
    """Interpolated microphone transfer function."""

    raw_freqs: npt.NDArray[np.float32]
    """1D array with frequencies of the measured transfer function in Hz."""

    raw_amps: npt.NDArray[np.float32]
    """1D array with amplitudes of the mic transfer function in DFS/µPa."""

    raw_phases: npt.NDArray[np.float32]
    """1D array with phases of the mic transfer function in radians."""

    def __init__(
        self,
        abs_calib: AbsCalibData,
        trans_fun: MicroTransferFunData,
        log: Logger | None = None,
    ) -> None:

        self.raw_freqs = np.array(trans_fun['frequencies'], np.float32)
        self.raw_amps = np.array(trans_fun['amplitudes'], np.float32)
        # Convert to DFS / µPa
        self.raw_amps /= abs_calib['sensitivity']
        self.raw_phases = np.array(trans_fun['phases'], np.float32)

        super().__init__(log)

    def get_interp_transfer_function(
        self,
        frequencies_ip: npt.NDArray[np.float32] | None = None,
        num_samples: int | None = None,
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
        return self._interp_amp_phase(
            freqs, self.raw_freqs, self.raw_amps, self.raw_phases
        )

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

    raw_freqs: npt.NDArray[np.float32]
    """1D array with frequencies of the output transfer function in Hz."""

    raw_amps: npt.NDArray[np.float32]
    """2D array with amplitudes of the output transfer function in muPa/FS

    Each row contains the amplitudes for a single output channel, where the row
    index corresponds to the same index used in output_channels. The columns
    contain amplitudes corresponding to the frequencies specified in raw_freqs.

    """

    raw_phases: npt.NDArray[np.float32]
    """2D array with phases of the output transfer function in radians.

    Each row contains the phases for a single output channel, where the row
    index corresponds to the same index used in output_channels. The columns
    contain phase values corresponding to the frequencies specified in
    raw_freqs.
    """

    input_channels: list[int]
    """Input Channels the output calibration was performed on"""

    output_channels: list[int]
    """Output channel of the calibration"""

    date: str
    """Time stamp of output calibration."""

    def __init__(
        self, calib_data: SpeakerCalibData, log: Logger | None = None
    ) -> None:
        self.raw_freqs = np.array(calib_data['frequencies'], np.float32)
        self.raw_amps = np.array(calib_data['max_out'], np.float32)
        self.raw_phases = np.array(calib_data['phase'], np.float32)

        super().__init__(log)

        self.date = calib_data['date']
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
            self.raw_freqs,
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
                'Stimulus frequency %.2f Hz outside calibrated boundaries.', f
            )

        amp = self.get_interp_transfer_function(
            ch, np.asarray([f], np.float32)
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
