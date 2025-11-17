"""Classes providing the context for a measurement.

This module contains classes that provide parameters
and references to live plot and window instances required
by the corresponding recorder to control the measurements
and update the associated plots.
"""

from dataclasses import dataclass

# from typing import Generic, TypeVar

from pyoae.anim import MsrmtFuncAnimation
from pyoae.calib import MicroTransferFunction


@dataclass
class MsrmtContext:
    """Parameters and instances to control measurement updates."""

    fs: float
    """Sampling frequency in Hz."""

    block_size: int
    """Number of samples in each block."""

    input_trans_fun: MicroTransferFunction | None
    """Handle to microphone transfer function.

    A microphone transfer function is used to correct the
    recorded signal from the microphone characteristics.

    Note:
        This is a dummy object for future implementation.
    """

    artifact_rejection_thr: float
    """Threshold for simple artifact rejection.

    Reject blocks with a root-mean-square (RMS) value
    exceeding ARTIFACT_REJECTION_THR * median_rms.
    """

    non_interactive: bool
    """Flag enabling/disabling non-interactive measurement mode."""

    msrmt_anim: MsrmtFuncAnimation | None
    """Animation instance for online display of measurement data."""


@dataclass
class DpoaeMsrmtContext(MsrmtContext):
    """Measurement context for continuous/pulsed DPOAE acquisition."""

    f1: float
    """Stimulus frequency in Hz of the first primary tone.

    Typically, f2/f1 = 1.2.
    """

    f2: float
    """Stimulus frequency in Hz of the second primary tone."""

    num_recorded_blocks: int
    """Number of recorded blocks."""
