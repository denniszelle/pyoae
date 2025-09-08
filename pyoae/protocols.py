"""Classes and functions to manage measurement protocols."""

from typing import TypedDict


class MsrmtParams(TypedDict):
    """Base class with general measurement parameters."""

    block_duration: float
    """Duration of a single measurement block in seconds.

    This also determines the frequency resolution in the measurement.
    """

    num_averaging_blocks: int
    """Number of measurement blocks acquired for averaging."""


class DpoaeMsrmtParams(MsrmtParams):
    """Measurement parameters for a single DPOAE measurement."""
    f2: float
    level2: float
    f1: float | None
    level1: float | None
    f2f1_ratio: float | None


class SoaeMsrmtParams(MsrmtParams):
    """Measurement parameters for SOAE measurement."""
    artifact_rejection_thresh: float


class CalibMsrmtParams(MsrmtParams):
    """Measurement parameters for multi-tone measurements."""
    f_start: float
    f_stop: float
    lines_per_octave: float
    amplitude_per_line: float
    num_channels: int


def get_default_soae_msrmt_params() -> SoaeMsrmtParams:
    """Returns default SOAE measurement parameters."""
    d: SoaeMsrmtParams = {
        'block_duration': 1.0,
        'num_averaging_blocks': 15,
        'artifact_rejection_thresh': 1.8
    }
    return d


def get_default_calib_msrmt_params() -> CalibMsrmtParams:
    """Returns default parameters for output calibration."""
    d: CalibMsrmtParams = {
        'block_duration': 1.0,
        'num_averaging_blocks': 1,
        'f_start': 200.0,
        'f_stop': 10000.0,
        'lines_per_octave': 9.1,
        'amplitude_per_line': 0.004,
        'num_channels': 2
    }
    return d
