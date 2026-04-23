"""Module providing functionality for storing calibration data."""

from typing import TypedDict


class AbsCalibData(TypedDict):
    """Container for absolute-calibration data in calibration file."""

    date: str
    """Date of the absolute calibration"""

    ref_frequency: float
    """Reference frequency of the absolute calibration"""

    sensitivity: float
    """Sensitivity of the microphone at the reference frequency"""

    calib_type: int
    """Identifier of the calibration type."""


class EarSimTransferFunData(TypedDict):
    """Container for ear simulator transfer-function data in file."""

    date: str
    """Date of the ear simulator transfer function calibration"""

    frequencies: list[float]
    """Frequencies of the transfer function"""

    amplitudes: list[float]
    """Amplitudes of the transfer function"""

    phases: list[float]
    """Phases of the transfer function"""


class EarSimCalibData(TypedDict):

    doc_type: str
    """Document type of the loaded calibration"""

    rev: int
    """Revision number of the document"""

    model: str
    """Model identifier of the probe"""

    transfer_function: EarSimTransferFunData
    """Ear simulator transfer function calibration of the probe"""


class MicroTransferFunData(TypedDict):
    """Container for transfer-function data in calibration file."""

    date: str
    """Date of the microphone transfer function calibration"""

    frequencies: list[float]
    """Frequencies for which the transfer function has been evaluated"""

    amplitudes: list[float]
    """Amplitudes of the transfer function"""

    phases: list[float]
    """Phase values of the transfer function"""


class MicroCalibData(TypedDict):
    """Container to load a microphone calibration file."""

    doc_type: str
    """Document type the loaded calibration"""

    rev: int
    """Revision number of the document"""

    probe_sn: str
    """Serial number of the probe"""

    model: str
    """Model identifier of the probe"""

    side: str
    """Measurement side for the probe"""

    abs_calibration: AbsCalibData
    """Sensitivity calibration of the probe"""

    transfer_function: MicroTransferFunData
    """Transfer function calibration of the probe"""


class SpeakerCalibData(TypedDict):
    """Container to save/load speaker calibration data.

    Container is compatible with JSON export.
    """

    date: str
    """Date the output calibration was performed"""

    output_channels: list[int]
    """Channels the output calibration was performed on"""

    input_channels: list[int]
    """Channels the output calibration was performed on"""

    frequencies: list[float]
    """Measured frequencies of the multitones"""

    max_out: list[list[float]]
    """Maximum amplitude output"""

    phase: list[list[float]]
    """Phase shift of the first given channel"""


def get_empty_micro_calib_data() -> MicroCalibData:
    """Returns an empty container for microphone-calibration data."""
    a: AbsCalibData = {
        'date': '',
        'ref_frequency': 1000.0,
        'sensitivity': 1,
        'calib_type': 2,
    }
    t: MicroTransferFunData = {
        'date': '',
        'frequencies': [1.0, 20000.0],
        'amplitudes': [1.0, 1.0],
        'phases': [0.0, 0.0],
    }
    d: MicroCalibData = {
        'doc_type': '',
        'rev': 2,
        'probe_sn': '',
        'model': '',
        'side': '',
        'abs_calibration': a,
        'transfer_function': t,
    }
    return d

def get_empty_ear_sim_calib() -> EarSimCalibData:
    """Returns an empty container for microphone-calibration data."""
    t: EarSimTransferFunData = {
        'date': '',
        'frequencies': [1.0, 20000.0],
        'amplitudes': [1.0, 1.0],
        'phases': [0.0, 0.0]
    }
    d: EarSimCalibData = {
        'doc_type': '',
        'rev': 1,
        'model': '',
        'transfer_function': t
    }
    return d


def get_empty_speaker_calib_data() -> SpeakerCalibData:
    """Returns an empty container for speaker-calibration data."""
    d: SpeakerCalibData = {
        'date': '',
        'output_channels': [],
        'input_channels': [],
        'frequencies': [],
        'max_out': [],
        'phase': [],
    }
    return d
