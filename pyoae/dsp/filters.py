"""Module with functions to implement filters."""

from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.signal as sig

HP_ORDER = 1201
"""Order of the high-pass filter"""

BP_ORDER = 1201
"""Order of the band-pass filter"""

REFERENCE_SAMPLING_RATE = 96000
"""Reference sampling frequency for filter order."""

RAMP_DURATION = 2
"""Duration of the ramp applied to averaged signal in ms"""


def scale_filter_order(order: int, fs: float) -> int:
    """Scale filter order to sampling frequency"""

    if order < 1:
        raise ValueError("Filter order must be >= 1")

    scaled_order = order * fs / REFERENCE_SAMPLING_RATE
    new_order = int(round(scaled_order))

    if (new_order - order) % 2 != 0:
        new_order += 1 if new_order < scaled_order else -1

    return max(1, new_order)


def high_pass_filter(
    y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    num_taps: int,
    samplerate: float,
    cutoff_hz: float = 200.0,
) -> npt.NDArray[np.float64]:
    """FIR high-pass with causal phase; same length as input."""
    if num_taps % 2 == 0:
        # HP linear-phase FIRs should use odd taps for Type-I symmetry
        num_taps += 1
    b = sig.firwin(
        num_taps,
        cutoff_hz,
        pass_zero="highpass",  # type: ignore
        fs=samplerate
    )

    D = (num_taps - 1) // 2  # group delay
    y_ext = np.pad(y.astype(np.float64, copy=False), (0, D), mode="edge")
    y_f = cast(np.ndarray, sig.lfilter(b, 1.0, y_ext))
    return y_f[D:D + y.shape[0]]


def low_pass_filter(
    y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    num_taps: int,
    samplerate: float,
    cutoff_hz: float = 200.0,
) -> npt.NDArray[np.float64]:
    """FIR low-pass with causal phase; same length as input."""
    if num_taps % 2 == 0:
        # HP linear-phase FIRs should use odd taps for Type-I symmetry
        num_taps += 1
    b = sig.firwin(
        num_taps,
        cutoff_hz,
        pass_zero="lowpass",  # type: ignore
        fs=samplerate
    )

    D = (num_taps - 1) // 2  # group delay
    y_ext = np.pad(y.astype(np.float64, copy=False), (0, D), mode="edge")
    y_f = cast(np.ndarray, sig.lfilter(b, 1.0, y_ext))
    return y_f[D:D + y.shape[0]]


def bp_pass_filter(
    y: npt.NDArray[np.float64],
    num_taps: int,
    samplerate: float,
    cutoff_hz: npt.NDArray,
    ramp_size: int = 0
) -> npt.NDArray[np.float64]:
    """FIR high-pass with causal phase; same length as input."""
    if num_taps % 2 == 0:
        # HP linear-phase FIRs should use odd taps for Type-I symmetry
        num_taps += 1
    b = sig.firwin(
        num_taps,
        cutoff_hz,
        pass_zero="bandpass",  # type: ignore
        fs=samplerate
    )

    D = (num_taps - 1) // 2  # group delay
    if ramp_size > 0:
        num_samples = len(y)
        if 2 * ramp_size > num_samples:
            raise ValueError(
                "ramp_size is too large: 2 * ramp_size must be <= number of samples."
            )
        t = np.linspace(0.0, np.pi, ramp_size, endpoint=True)
        rise = 0.5 - 0.5 * np.cos(t)
        fall = rise[::-1]
        window = np.ones(num_samples, dtype=float)
        window[:ramp_size] *= rise
        window[-ramp_size:] *= fall
        y_w = y * window
    else:
        y_w = y

    y_ext = np.pad(y_w.astype(np.float64, copy=False), (0, D), mode="edge")
    y_f = cast(np.ndarray, sig.lfilter(b, 1.0, y_ext))
    return y_f[D:D + y.shape[0]]
