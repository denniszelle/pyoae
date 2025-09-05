"""Functions to generate output signals."""

from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import windows

from pyoae import converter
from pyoae.calib import OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.protocols import DpoaeMsrmtParams
from pyoae.signals import PeriodicRampSignal


SYNC_CROSS: Final[int] = 10
"""Number of minimum zero crossings in sync signal."""

SYNC_AMPLITUDE: Final[float] = 0.1
"""Amplitude of the sync signal in digital full scale."""

SYNC_FREQUENCY: Final[float] = 4000
"""Carrier frequency of the sync pulse in Hz."""


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


def calculate_cdpoae_frequencies(
    msrmt_params: DpoaeMsrmtParams,
    block_duration: float
) -> tuple[float, float]:
    """Calculate primary-tone frequencies f1 and f2 for cDPOAE acquisition."""
    f2 = correct_frequency(msrmt_params['f2'], block_duration)
    if msrmt_params['f1'] is None:
        if msrmt_params['f2f1_ratio'] is None:
            f1 = 0.0  # invalid stimulus parameters
            # TODO: log error
        else:
            f1 = f2/msrmt_params['f2f1_ratio']
            f1 = correct_frequency(f1, block_duration)
    else:
        f1 = correct_frequency(msrmt_params['f1'], block_duration)

    print(
        'Setting primary-tone frequencies: '
        f'f1: {f1:.2f} Hz, f2: {f2:.2f} Hz, '
        f'(f2/f1 = {f2/f1: .3f})'
    )
    return (f1, f2)


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
    level1: float | None
) -> tuple[float, float]:
    """Calculates peak pressure amplitudes from dB SPL levels."""
    if level1 is None:
        # as first approximation, use Kummer et al. 1998
        # TODO: add other rules
        level1 = 0.4 * level2 + 39

    amplitude1 = converter.db_spl_to_peak_mupa(level1)
    amplitude2 = converter.db_spl_to_peak_mupa(level2)
    print(
        'Setting output pressures for DPOAE acquisition: '
        f'L1: {level1:.1f} dB SPL ({amplitude1:.5f} re FS).'
        f'L2: {level2:.1f} dBFS ({amplitude2:.5f} re FS).'
    )
    return (amplitude1, amplitude2)



def generate_sync(fs: float) -> npt.NDArray[np.float32]:
    """Generates and returns the sync signal.

    Args:
        fs: Sampling frequency in Hz

    Returns:
        sync_pulse: 1D array of sync pulse signal
    """
    k = np.ceil(fs / (4.0 * SYNC_FREQUENCY))
    samples_per_sync_period = 4.0 * k
    f_sync = fs / samples_per_sync_period
    p = np.ceil((SYNC_CROSS + 1) / 2)
    t_off = (p / f_sync) - 1/fs
    num_samples = int(t_off * fs)
    t = np.arange(num_samples) / fs
    y = np.sin(2 * np.pi * f_sync * t)
    w = windows.tukey(num_samples)
    sync_pulse = SYNC_AMPLITUDE * w * y
    return sync_pulse


def generate_cdpoae_stimuli(
    msrmt_params: DpoaeMsrmtParams,
    output_calibration: OutputCalibration | None = None
) -> tuple[PeriodicRampSignal, PeriodicRampSignal]:
    """Generates primary tones for continuous DPOAE acquisition."""

    # calculate number of samples and ensure an integer number
    num_block_samples = int(
        DeviceConfig.sample_rate * msrmt_params['block_duration']
    )
    # ensure block duration matches number of block samples
    block_duration = num_block_samples / DeviceConfig.sample_rate
    if block_duration != msrmt_params['block_duration']:
        print(
            f'Block duration adjusted to {block_duration*1E3:.2f} ms'
        )

    f1, f2 = calculate_cdpoae_frequencies(msrmt_params, block_duration)

    if output_calibration is None:
        # No calibration for output channels available.
        amplitude1, amplitude2 = calculate_full_scale_amplitudes(
            msrmt_params['level2'],
            msrmt_params['level1']
        )
    else:
        pressure1, pressure2 = calculate_pressure_amplitudes(
            msrmt_params['level2'],
            msrmt_params['level1']
        )
        amplitude1 = output_calibration.pressure_to_full_scale(
            0,
            pressure1,
            f1
        )
        amplitude2 = output_calibration.pressure_to_full_scale(
            1,
            pressure2,
            f2
        )
        print(
            'Setting output amplitudes for DPOAE acquisition: '
            f'p1: {pressure1:.1f} muPa ({amplitude1:.5f} re FS).'
            f'p2: {pressure2:.1f} muPa ({amplitude2:.5f} re FS).'
        )


    # Generate output signals
    samples = np.arange(num_block_samples, dtype=np.float32)
    t = samples / DeviceConfig.sample_rate
    stimulus1 = amplitude1 * np.sin(2*np.pi*f1*t).astype(np.float32)
    stimulus2 = amplitude2 * np.sin(2*np.pi*f2*t).astype(np.float32)

    num_total_recording_samples = (
        num_block_samples * msrmt_params['num_averaging_blocks']
    )

    # we always use rising and falling edges
    ramp_len = int(
        DeviceConfig.ramp_duration * 1E-3 * DeviceConfig.sample_rate
    )
    ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
    ramp = ramp.astype(np.float32)

    signal1 = PeriodicRampSignal(
        stimulus1,
        num_total_recording_samples,
        ramp
    )
    signal2 = PeriodicRampSignal(
        stimulus2,
        num_total_recording_samples,
        ramp
    )
    return (signal1, signal2)
