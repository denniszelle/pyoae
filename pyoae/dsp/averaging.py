"""Module with various averaging functions in time and spectral domain.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.signal

from pyoae.dsp.opt_avg import OptAverage


def welch_spectrum(
    x: npt.NDArray[np.float32],
    fs: float,
    window: str = 'hann',
    window_samples: int = 256,
    overlap_samples: int | None = None,
    detrend: Literal['linear', 'constant'] = 'constant',
    use_opt_avg: bool = True
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Estimates a spectrum using Welch's method.

    Args:
        x: Float array as time-domain signal
        fs: Sampling frequency of the measurement
        window: Type of the window (see scipy.signal.get_window)
        window_samples: Samples for each window
        overlap_samples: Window overlap samples, window_samples/2 by default
        detrend: Remove linear or constant trend beforehand
        use_opt_avg: Flag to enable/disable optimized averaging.

    Returns:
        tuple[frequencies, spec_avg]

        - **frequencies**: Float array containing the corresponding frequencies
            for the magnitudes.
        - **spec_avg**: Float array with averaged spectrum scaled
            according to the scaling input

    """
    if overlap_samples is None:
        overlap_samples = window_samples // 2
    step = window_samples - overlap_samples
    x_seg = np.lib.stride_tricks.sliding_window_view(
        x, window_shape=window_samples
    )[::step]  # (num_blocks, num_samples)

    if detrend:
        x_seg = scipy.signal.detrend(x_seg, type=detrend, axis=-1)

    win = scipy.signal.get_window(window, window_samples)
    x_seg = x_seg * win
    x_fft = np.fft.rfft(x_seg, n=window_samples, axis=-1)

    # Magnitude with coherent-gain correction
    spectrum = np.abs(x_fft)

    # One-sided amplitude scaling: double only non-DC/non-Nyquist bins
    if window_samples % 2 == 0:
        spectrum[..., 1:-1] *= 2.0
    else:
        spectrum[..., 1:] *= 2.0

    spectrum /= np.sum(win)  # coherent gain
    frequencies = np.fft.rfftfreq(window_samples, 1/fs)

    if use_opt_avg:
        # Obtain RMS values of spectrum
        rms_vals = np.sqrt(np.mean(np.square(spectrum), axis=1))

        # Apply optimized averaging

        opt_averager = OptAverage()
        opt_averager.setup(len(rms_vals))
        for i, rms_val_i in enumerate(rms_vals):
            opt_averager.set_noise_value(i, rms_val_i)
        opt_averager.evaluate_averaging()

        accepted_idc = opt_averager.accepted_idx
    else:
        accepted_idc = np.empty(0, dtype=np.int_)

    if len(accepted_idc):
        spec_avg = spectrum[accepted_idc].mean(axis=0)
    else:
        spec_avg = np.zeros(len(spectrum[0]))

    return frequencies.astype(np.float32), spec_avg.astype(np.float32)


def reshape_recording(
    signal: npt.NDArray[np.floating],
    block_size: int,
) -> np.ndarray:
    """Reshapes a 1D recorded signal into measurement blocks.

    Args:
        signal: 1D array containing the recorded signal (float32 or float64).
        block_size: Number of samples per acquisition block.

    Returns:
        2D array of shape (num_blocks, block_size) containing the
        measurement blocks. Excess samples that do not fill a
        complete block are discarded.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    num_blocks = signal.shape[0] // block_size
    num_used_samples = num_blocks * block_size
    return signal[:num_used_samples].reshape(num_blocks, block_size)


def calculate_ptpv_ensembles(
    signal: npt.NDArray[np.floating],
    block_size: int,
    num_ptpv_phases: int = 4,
) -> npt.NDArray[np.floating]:
    """Forms PTPV ensembles by averaging consecutive phase-shifted blocks.

    The input signal is interpreted as a sequence of acquisition blocks of
    length `block_size`. PTPV ensembles are formed by averaging
    `num_ptpv_phases` consecutive blocks (one full PTPV phase cycle), which
    cancels the primary tones due to their phase rotations.

    Reference:
        Whitehead et al. (1996)

    Args:
        signal:
            1D array containing the recorded signal (float32 or float64).
        block_size:
            Number of samples per acquisition block.
        num_ptpv_phases:
            Number of distinct primary-tone phase shifts per PTPV cycle.

    Returns:
        2D array of shape (num_ensembles, block_size) containing the PTPV ensembles.
        Excess samples that do not fill a complete block or a complete PTPV cycle
        are discarded.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if num_ptpv_phases <= 0:
        raise ValueError("num_ptpv_phases must be a positive integer")

    num_total_blocks = signal.shape[0] // block_size
    num_ensembles = num_total_blocks // num_ptpv_phases
    if num_ensembles == 0:
        # Return empty array with correct shape/dtype
        return np.empty((0, block_size), dtype=signal.dtype)

    num_used_blocks = num_ensembles * num_ptpv_phases
    num_used_samples = num_used_blocks * block_size

    blocks = signal[:num_used_samples].reshape(num_used_blocks, block_size)
    ensembles = blocks.reshape(
        num_ensembles,
        num_ptpv_phases,
        block_size
    ).mean(axis=1)

    return ensembles


def split_ptpv_segments(
    blocks: np.ndarray,
    num_ptpv_phases: int = 4,
) -> list[np.ndarray]:
    """Group interleaved recording blocks into per-phase PTPV segments.

    The input is assumed to be an interleaved sequence of blocks acquired with
    cyclic primary-tone phase shifts, e.g. for 4 shifts:

        phase0, phase1, phase2, phase3, phase0, phase1, ...

    This function de-interleaves the sequence into `num_ptpv_phases` arrays,
    one per phase shift.

    Args:
        blocks: Array of shape (n_blocks, n_samples) containing
          consecutive blocks in acquisition order.
        num_ptpv_phases: Number of distinct primary-tone phase
          shifts (PTPV segments).

    Returns:
        List of length `num_phase_shifts`. Each element is an array of shape
          (n_blocks_i, n_samples) containing the blocks for that phase shift.
          Note that `n_blocks_i` may differ by at most 1 between phases if the
          total number of blocks is not divisible by `num_ptpv_phases`.
    """
    if blocks.ndim != 2:
        raise ValueError(
            f"Expected blocks with shape (n_blocks, n_samples), got {blocks.shape}"
        )
    if num_ptpv_phases <= 0:
        raise ValueError("num_ptpv_phases must be a positive integer")

    return [blocks[i::num_ptpv_phases] for i in range(num_ptpv_phases)]


def calculate_optimized_ptpv_average(
    average_calculators: list[OptAverage],
    phase_segments: list[np.ndarray],
    num_segments: int | None = None,
) -> np.ndarray:
    """Computes a PTPV-cancelled average from per-phase accepted segments.

    This function assumes the input has already been grouped into
    PTPV phase segments. For each phase segment `i`, it averages only
    the blocks whose indices are marked as accepted by
    `average_calculators[i]`. `accepted_idx` is assumed to be sorted
    from lowest to highest noise (best→worst). The resulting per-phase
    sub-averages are then averaged across all PTPV phases to cancel
    the primary tones (PTPV).

    Reference:
        Whitehead et al. (1996)

    Note:
        This approach can yield an unequal number of accepted
        segments per phase, increasing the overall acceptance rate,
        but potentially weighting phases asymmetrically. In contrast,
        forming fixed-size PTPV ensembles first (one segment per phase)
        provides symmetric phase contributions, but a single rejected segment
        can invalidate the whole ensemble.

    Args:
        average_calculators: One `OptAverage` instance per PTPV phase segment.
          Each instance must provide `accepted_idx`, indexing into
          the segment axis of `phase_segment[i]`.
        phase_segments: List of arrays, one per PTPV phase segment. Each array
          must have shape (num_segments, num_samples).
        num_segments: If given, use only the first `num_segments` accepted indices
          per phase (i.e., `accepted_idx[:num_segments]`). If None,
          use all accepted segments.

    Returns:
        1D NumPy array of shape (n_samples,) containing the
          PTPV-averaged signal.
    """
    num_segments = len(phase_segments)
    if len(average_calculators) != num_segments:
        raise ValueError(
            f"Expected {num_segments} average_calculators, got {len(average_calculators)}"
        )

    per_phase_avg: list[np.ndarray] = []
    for i in range(num_segments):
        segments = phase_segments[i]
        accepted_idx = average_calculators[i].accepted_idx
        if num_segments is not None:
            accepted_idx = accepted_idx[:num_segments]

        per_phase_avg.append(np.mean(segments[accepted_idx, :], axis=0))

    return np.mean(np.stack(per_phase_avg, axis=0), axis=0)
