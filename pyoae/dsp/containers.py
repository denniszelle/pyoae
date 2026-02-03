"""Module with data containers for typing"""

from typing import TypedDict

import numpy as np
import numpy.typing as npt


class DpoaeMsrmtData(TypedDict):
    """A general container for raw DPOAE measurement data."""

    recorded_signal: npt.NDArray[np.float32]
    """Raw recorded time signal"""

    samplerate: float
    """Digital sample rate of the input"""

    f1: float
    """Spectral frequency of the first stimulus"""

    f2: float
    """Spectral frequency of the second stimulus"""

    level1: float
    """Level of the first stimulus"""

    level2: float
    """Level of the second stimulus"""

    num_block_samples: int
    """Number of samples per block"""

    recorded_sync: npt.NDArray[np.float32]
    """Raw signal of the sync signal"""

    out_ch: list[int] | None
    """Output channels used"""

    in_ch: int | None
    """Input channel used"""

    msrmt_idx: int | None
    """Measurement index corresponding to position in protocol"""


class ContDpoaeRecording(TypedDict):
    """Structured content from a DPOAE recording file."""

    recording: DpoaeMsrmtData
    """Data of continuos DPOAE recording"""

    average: npt.NDArray[np.float64] | None
    """Averaged time signal"""

    spectrum: npt.NDArray[np.float64] | None
    """Averaged spectral signal"""


class PulseDpoaeRecording(TypedDict):
    """Structured content from a pulsed DPOAE recording file."""

    recording: DpoaeMsrmtData
    """Data of pulsed DPOAE recording"""

    average: npt.NDArray[np.float64] | None
    """Average raw data time signal"""

    signal: npt.NDArray[np.float64] | None
    """Average time signal"""
