"""Module with general helper functions for PyOAE."""

import re

from pyoae.calib_storage import SpeakerCalibData, EarSimTransferFunData
from pyoae.calib_transfer import EarSimTransferFunction


def sanitize_filename_part(part: str) -> str:
    """Remove characters invalid in filenames and strip whitespace."""
    # Invalid on Windows: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]', "", part).strip()


def extract_ear_sim_data(
    calib_data: SpeakerCalibData,
    output_channels: list[int],
) -> list[EarSimTransferFunction] | None:
    """Extract ordered ear simulator data from calibration data."""

    # Order to match microphone transfer functions.
    if (
        calib_data['ear_sim_frequencies'] is None
        or calib_data['ear_sim_amplitudes'] is None
        or calib_data['ear_sim_phases'] is None
    ):
        # TODO: Change to logging
        print(f'No ear simulator data available')
        return

    channel_to_mic = dict(
        zip(
            calib_data['output_channels'],
            calib_data['input_channels']
        )
    )
    hw_order = list(dict.fromkeys(calib_data['input_channels']))
    hw_to_norm = {hw: i for i, hw in enumerate(hw_order)}
    ear_sim_tfs = []
    seen = []
    for ch in output_channels:
        hw_mic = channel_to_mic[ch]
        norm_mic = hw_to_norm[hw_mic]
        if norm_mic not in seen:
            seen.append(norm_mic)
            data: EarSimTransferFunData = {
                'date':'',
                'frequencies': calib_data['ear_sim_frequencies'][norm_mic],
                'amplitudes': calib_data['ear_sim_amplitudes'][norm_mic],
                'phases': calib_data['ear_sim_phases'][norm_mic]
            }
            ear_sim_tfs.append(EarSimTransferFunction(data))

    return ear_sim_tfs
