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


def get_default_soae_msrmt_params() -> SoaeMsrmtParams:
    """Returns default SOAE measurement parameters."""
    d: SoaeMsrmtParams = {
        'block_duration': 1.0,
        'num_averaging_blocks': 15,
        'artifact_rejection_thresh': 1.8
    }
    return d
