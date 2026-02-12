"""Classes and functions to manage measurement protocols."""

from typing import (
    Any,
    TypedDict,
    Final
)

import numpy as np
import numpy.typing as npt


OUTPUT_CALIB_KEYS: Final[list[str]] = [
    'block_duration',
    'num_averaging_blocks',
    'num_clusters',
    'f_start',
    'f_stop',
    'lines_per_octave',
    'amplitude_per_line'
]
"""Necessary keys for output calibration"""


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


class PulseStimulus(TypedDict):
    """Time markers defining a pulsed stimulus."""
    t_rise: float
    t_fall: float
    t_on: float
    duration: float
    is_short_pulse: bool


class PulseDpoaeMsrmtParams(DpoaeMsrmtParams):
    """Measurement parameters for a single pulse DPOAE measurement."""
    f1_pulse: PulseStimulus
    f2_pulse: PulseStimulus


class CalibMsrmtParams(MsrmtParams):
    """Measurement parameters for multi-tone measurements."""

    num_clusters: int
    """Number of clusters the spectral tones are distributed on."""

    f_start: float
    """Start frequency"""

    f_stop: float
    """Stop frequency"""

    lines_per_octave: float
    """Spectral signals per octave"""

    amplitude_per_line: float
    """Amplitude of each spectral signal"""


class CalibMsrmtDef(MsrmtParams):
    """Measurement definition for custom multi-tone protocols"""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the multi-tones"""

    phases: npt.NDArray[np.float32]
    """Phases of the multi-tones"""

    amplitudes: npt.NDArray[np.float32]
    """Amplitudes of the multi-tones"""

    cluster_idc: npt.NDArray[np.int32]
    """Cluster indices of the multi tones"""


def get_default_soae_msrmt_params() -> MsrmtParams:
    """Returns default SOAE measurement parameters."""
    d: MsrmtParams = {
        'block_duration': 1.0,
        'num_averaging_blocks': 15
    }
    return d


def get_default_calib_msrmt_params() -> CalibMsrmtParams:
    """Returns default parameters for output calibration."""
    d: CalibMsrmtParams = {
        'block_duration': 1.0,
        'num_averaging_blocks': 1,
        'num_clusters': 1,
        'f_start': 200.0,
        'f_stop': 10000.0,
        'lines_per_octave': 9.1,
        'amplitude_per_line': 0.004
    }
    return d


def get_custom_calib_msrmt_params(
    calib_params: dict[str, Any]
) -> CalibMsrmtParams | None:
    """Returns file-loaded parameters for output calibration."""

    for key_i in OUTPUT_CALIB_KEYS:
        if key_i not in calib_params.keys():
            return None

    d: CalibMsrmtParams = {
        'block_duration': calib_params['block_duration'],
        'num_averaging_blocks': calib_params['num_averaging_blocks'],
        'num_clusters': calib_params['num_clusters'],
        'f_start': calib_params['f_start'],
        'f_stop': calib_params['f_stop'],
        'lines_per_octave': calib_params['lines_per_octave'],
        'amplitude_per_line': calib_params['amplitude_per_line']
    }
    return d
