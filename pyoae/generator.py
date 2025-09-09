"""Functions to generate output signals."""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import windows

from pyoae import converter
from pyoae.calib import OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.protocols import DpoaeMsrmtParams
# from pyoae.signals import PeriodicRampSignal


SYNC_CROSS: Final[int] = 10
"""Number of minimum zero crossings in sync signal."""

SYNC_AMPLITUDE: Final[float] = 0.2
"""Amplitude of the sync signal in digital full scale."""

SYNC_FREQUENCY: Final[float] = 4000
"""Carrier frequency of the sync pulse in Hz."""


@dataclass
class DpoaeStimulus:
    """Container for DPOAE stimulus parameters."""

    f1: float
    """Frequency of first primary tone in Hz."""

    f2: float
    """Frequency of second primary tone in Hz."""

    level1: float
    """Stimulus level of first primary tone in dB SPL."""

    level2: float
    """Stimulus level of second primary tone in dB SPL."""

    def calculate_cdpoae_frequencies(
        self,
        msrmt_params: DpoaeMsrmtParams,
        block_duration: float
    ) -> tuple[float, float]:
        """Calculate primary-tone frequencies f1 and f2 for cDPOAE acquisition."""
        self.f2 = correct_frequency(msrmt_params['f2'], block_duration)
        if msrmt_params['f1'] is None:
            if msrmt_params['f2f1_ratio'] is None:
                self.f1 = 0.0  # invalid stimulus parameters
                # TODO: log error
            else:
                self.f1 = self.f2/msrmt_params['f2f1_ratio']
                self.f1 = correct_frequency(self.f1, block_duration)
        else:
            self.f1 = correct_frequency(msrmt_params['f1'], block_duration)

        print(
            'Setting primary-tone frequencies: '
            f'f1: {self.f1:.2f} Hz, f2: {self.f2:.2f} Hz, '
            f'(f2/f1 = {self.f2/self.f1: .3f})'
        )
        return (self.f1, self.f2)

    def generate_cdpoae_stimuli(
        self,
        num_block_samples: int,
        output_calibration: OutputCalibration | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Generates primary tones for continuous DPOAE acquisition."""

        if output_calibration is None:
            # No calibration for output channels available.
            amplitude1, amplitude2 = calculate_full_scale_amplitudes(
                self.level2,
                self.level1
            )
        else:
            pressure1, pressure2 = calculate_pressure_amplitudes(
                self.level2,
                self.level1
            )
            amplitude1 = output_calibration.pressure_to_full_scale(
                0,
                pressure1,
                self.f1
            )
            amplitude2 = output_calibration.pressure_to_full_scale(
                1,
                pressure2,
                self.f2
            )
            print(
                'Setting output amplitudes for DPOAE acquisition: '
                f'p1: {pressure1:.1f} muPa ({amplitude1:.5f} re FS).'
                f'p2: {pressure2:.1f} muPa ({amplitude2:.5f} re FS).'
            )

        # Generate output signals
        samples = np.arange(num_block_samples, dtype=np.float32)
        t = samples / DeviceConfig.sample_rate
        stimulus1 = amplitude1 * np.sin(2*np.pi*self.f1*t).astype(np.float32)
        stimulus2 = amplitude2 * np.sin(2*np.pi*self.f2*t).astype(np.float32)
        return (stimulus1, stimulus2)


def correct_frequency(frequency: float, block_duration: float) -> float:
    """Corrects frequency to obtain an integer number of periods.

    Args:
        frequency: Stimulus frequency in Hz.
        block_duration: Length of acquisition block in seconds.
          A block represents a time segment that is repeatedly
          presented and used for averaging.
    """
    periods = block_duration*frequency
    periods = round(periods)
    return periods/block_duration


def calculate_pt1_level(msrmt_params: DpoaeMsrmtParams) -> float:
    """Calculates the stimulus level of the first primary tone."""
    if msrmt_params['level1'] is None:
        # as first approximation, use Kummer et al. 1998
        # TODO: add other rules
        level1 = 0.4 * msrmt_params['level2'] + 39
    else:
        level1 = msrmt_params['level1']
    return level1


def calculate_full_scale_amplitudes(
    level2: float, level1: float | None
) -> tuple[float, float]:
    """Computes digital full-scale amplitudes from dBFS levels.

    Uses levels as dBFS (0 dBFS = digital full scale, i.e. 1).
    Applies equal stimulus level rule (L1 = L2) if L1
    was not specified
    """
    level1 = level1 or level2
    amplitude1 = converter.db_to_lin(level1)
    amplitude2 = converter.db_to_lin(level2)
    amplitude1 = max(0.0, min(amplitude1, 1.0))
    amplitude2 = max(0.0, min(amplitude2, 1.0))
    print(
        'Setting output amplitudes for DPOAE acquisition: '
        f'L1: {level1:.1f} dBFS ({amplitude1:.5f} re FS).'
        f'L2: {level2:.1f} dBFS ({amplitude2:.5f} re FS).'
    )
    return (amplitude1, amplitude2)


def calculate_pressure_amplitudes(
    level2: float,
    level1: float
) -> tuple[float, float]:
    """Calculates peak pressure amplitudes from dB SPL levels."""
    amplitude1 = converter.db_spl_to_peak_mupa(level1)
    amplitude2 = converter.db_spl_to_peak_mupa(level2)
    print(
        'Setting output pressures for DPOAE acquisition: '
        f'L1: {level1:.1f} dB SPL ({amplitude1:.5f} re FS).'
        f'L2: {level2:.1f} dB SPL ({amplitude2:.5f} re FS).'
    )
    return (amplitude1, amplitude2)


def generate_sync(sample_rate: float) -> npt.NDArray[np.float32]:
    """Generates and returns the sync signal.

    Args:
        fs: Sampling frequency in Hz

    Returns:
        sync_pulse: 1D array of sync pulse signal
    """
    k = np.ceil(sample_rate / (4.0 * SYNC_FREQUENCY))
    samples_per_sync_period = 4.0 * k
    f_sync = sample_rate / samples_per_sync_period
    p = np.ceil((SYNC_CROSS + 1) / 2)
    t_off = (p / f_sync) - 1/sample_rate
    num_samples = int(t_off * sample_rate)
    t = np.arange(num_samples) / sample_rate
    y = np.sin(2 * np.pi * f_sync * t)
    w = windows.tukey(num_samples)
    sync_pulse = SYNC_AMPLITUDE * w * y
    return sync_pulse


def compute_mt_frequencies(
    f_start: float,
    f_stop: float,
    lines_per_octave: float
) -> npt.NDArray[np.floating]:
    """Computes multi-tone frequencies.

    Computes frequency lines from start to stop frequency adjusted to
    segment length with the specified lines per octave.
    """
    b = 2 ** (1/lines_per_octave)
    n = int(np.log(f_stop/f_start)/np.log(b))
    f = f_start * b ** (np.arange(n))
    f = np.round(f)
    return f


def compute_mt_phases(num_frequencies: int) -> npt.NDArray[np.floating]:
    """Computes approximately equally distributed phases"""
    phi = np.zeros(num_frequencies)
    for i in range(num_frequencies):
        phi[i] = np.random.uniform(0, 2 * np.pi)
    return phi


def interlace_mt_frequencies(
    frequencies: npt.NDArray[np.floating],
    num_tones: int
) -> list[npt.NDArray[np.floating]]:
    """Creates interlaced frequency lists.

    Splits frequency array into `num_tones` interlaced sets.
    Each set contains every nth frequency starting at a different offset.
    """
    return [frequencies[i::num_tones] for i in range(num_tones)]


def get_time_vector(
    num_samples: int,
    sample_rate: float
) -> npt.NDArray[np.float32]:
    """Computes and returns a time vector in seconds."""
    time_vec = np.arange(num_samples) / sample_rate
    return time_vec.astype(np.float32)


def generate_mt_signal(
    num_samples: int,
    sample_rate: float,
    frequencies: npt.NDArray[np.float32],
    phases: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Create a multi-tone signal used for output calibration."""
    time_vec = get_time_vector(num_samples, sample_rate)

    mt_signal = np.zeros_like(time_vec)
    for i, f in enumerate(frequencies):
        mt_signal += np.sin(2 * np.pi * f * time_vec + phases[i])

    return mt_signal
