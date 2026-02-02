"""Module with functions to process DPOAE data."""

import numpy as np
import numpy.typing as npt


def estimate_power(y: npt.NDArray[np.float32 | np.float64]) -> float:
    """Estimates the signal power as RMS for a given signal."""
    return np.sqrt(np.mean(np.square(y)))


def estimate_spectral_noise(
    y: npt.NDArray[np.floating],
    num_samples: int,
    f_signal: float,
    samplerate: float,
    noise_bin_offset: int = 10,
    num_noise_bins: int = 10
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

    # Build noise-bin indices on both sides, skipping close bins
    idx_left_start = idx_signal - noise_bin_offset - num_noise_bins
    idx_left_end = idx_signal - noise_bin_offset
    idx_right_start = idx_signal + noise_bin_offset
    idx_right_end = idx_signal + noise_bin_offset + num_noise_bins

    # Clip to valid range [0, len(mag_rms)-1] and drop empty sides gracefully
    n_bins = mag_rms.shape[0]
    left = np.arange(max(0, idx_left_start),  max(0, idx_left_end))
    right = np.arange(min(n_bins, idx_right_start), min(n_bins, idx_right_end))
    noise_bins = np.concatenate((left, right))
    if noise_bins.size == 0:
        raise ValueError("No valid noise bins (signal too close to edges).")

    # Power averaging for noise, then RMS
    noise_rms = float(np.sqrt(np.mean(mag_rms[noise_bins] ** 2)))

    return noise_rms
