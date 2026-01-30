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
