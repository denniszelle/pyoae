"""Classes providing the context for a measurement.

This module contains classes that provide parameters
and references to live plot and window instances required
by the corresponding recorder to control the measurements
and update the associated plots.
"""

from dataclasses import dataclass

from pyoae.calib_storage import MicroTransferFunction


@dataclass
class BaseMsrmtContext:
    """Parameters and instances to control measurement updates."""

    fs: float
    """Sampling frequency in Hz."""

    block_size: int
    """Number of samples in each block."""

    non_interactive: bool
    """Flag enabling/disabling non-interactive measurement mode."""


@dataclass
class MsrmtContext(BaseMsrmtContext):
    """Parameters and instances to control measurement updates."""

    input_trans_fun: list[MicroTransferFunction] | None
    """Handle to list of microphone transfer functions.

    A microphone transfer function is used to correct the
    recorded signal from the microphone characteristics.
    """


@dataclass
class DpoaeMsrmtContext(BaseMsrmtContext):
    """Measurement context for continuous/pulsed DPOAE acquisition."""

    input_trans_fun: MicroTransferFunction | None
    """Handle to microphone transfer function.

    A microphone transfer function is used to correct the
    recorded signal from the microphone characteristics.
    """

    f1: float
    """Stimulus frequency in Hz of the first primary tone.

    Typically, f2/f1 = 1.2.
    """

    f2: float
    """Stimulus frequency in Hz of the second primary tone."""

    num_recorded_blocks: int
    """Number of recorded blocks."""
