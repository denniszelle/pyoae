"""Module with noise-estimation functions."""

import numpy as np
import numpy.typing as npt

from pyoae.dsp import math


NUM_NOISE_BINS_PER_SIDE = 5
"""Number of noise bins close to DPOAE frequency."""

MIN_SNR = 10
"""Minimum signal-to-noise ratio in dB to accept a DPOAE."""


def cdpoae_noise_bins(
    idx_signal: int,
    num_max_bins: int,
    num_noise_bins: int = NUM_NOISE_BINS_PER_SIDE
) -> npt.NDArray[np.int_]:
    """Calculates the indices of noise bins for continuous signals."""
    if idx_signal < 0:
        err = f'Signal index {idx_signal} must be positive.'
        raise ValueError(err)

    if idx_signal > num_max_bins:
        err = f'Signal index {idx_signal} out of bounds ({num_max_bins}).'
        raise ValueError(err)

    idx_left_start = idx_signal - num_noise_bins
    idx_right_start = idx_signal + 1
    idx_right_end = idx_right_start + num_noise_bins

    left_bin_idx = np.arange(max(0, idx_left_start), idx_signal)
    right_bin_idx = np.arange(
        min(num_max_bins, idx_right_start),
        min(num_max_bins, idx_right_end)
    )
    noise_bins = np.concatenate([left_bin_idx, right_bin_idx])
    if noise_bins.size == 0:
        raise ValueError("No valid noise bins (signal too close to edges).")
    return noise_bins


def pdpoae_noise_bins(
    idx_signal: int,
    df: float,
    signal_bw: float,
    num_max_bins: int,
    num_noise_bins: int = NUM_NOISE_BINS_PER_SIDE
) -> npt.NDArray[np.int_]:
    """Calculates the indices of noise bins for pulsed signals."""
    if idx_signal < 0:
        raise ValueError('Signal index must be positive.')

    if idx_signal > num_max_bins:
        raise ValueError('Signal index out of bounds.')

    num_signal_bins = int(np.ceil(signal_bw / df))

    if num_signal_bins % 2 == 0:
        # ensure odd number of bins centered around idx_signal
        num_signal_bins += 1

    num_signal_side_bins = int(0.5 * (num_signal_bins - 1))
    signal_left_bnd = idx_signal - num_signal_side_bins
    signal_right_bnd = idx_signal + num_signal_side_bins + 1

    idx_left_start = signal_left_bnd - num_noise_bins
    idx_right_start = signal_right_bnd + 1
    idx_right_end = idx_right_start + num_noise_bins


    left_bin_idx = np.arange(
        max(0, idx_left_start),
        max(0, signal_left_bnd)
    )
    right_bin_idx = np.arange(
        min(num_max_bins, idx_right_start),
        min(num_max_bins, idx_right_end)
    )

    return np.concatenate([left_bin_idx, right_bin_idx])


def estimate_cdpoae_spectral_noise(
    y: npt.NDArray[np.floating],
    num_samples: int,
    f_signal: float,
    samplerate: float
) -> float:
    """Estimate narrow-band noise around a harmonic's bin as RMS amplitude.

    Returns:
        Noise RMS (same units as y's amplitude; per-bin, not per Hz).
    """
    # Detrend (remove DC) to reduce leakage into neighbors
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()

    # FFT (one-sided), explicit size
    Y = np.fft.rfft(y, n=num_samples)

    # Convert to one-sided **peak** amplitude per bin
    # A_peak = 2*|Y|/N for 0<k<N/2; DC and Nyquist are not doubled
    mag = np.abs(Y) / num_samples
    if num_samples % 2 == 0:
        # even N: rfft has N/2+1 bins, last is Nyquist
        mag[1:-1] *= 2.0
    else:
        # odd N: last bin is not Nyquist; all bins except DC are doubled
        mag[1:] *= 2.0

    # Convert to **RMS** per bin (sine RMS = peak / sqrt(2))
    mag_rms = mag / np.sqrt(2.0)

    # Locate signal bin (coherent sampling assumed)
    # Since we're ensuring that continuous primary tones
    # exhibit an integer number of periods within an
    # acquisition block, we're save to use the following
    # formula.
    idx_signal = int(round(f_signal * num_samples / samplerate))
    # Alternatively search for the closest bin using the fft
    # frequencies:
    # frequencies = np.fft.rfftfreq(num_samples, d=1/samplerate)
    # idx_signal = np.argmin(np.abs(frequencies - f_signal))

    noise_bins = cdpoae_noise_bins(idx_signal, mag_rms.shape[0])
    return math.rms(mag_rms[noise_bins])
