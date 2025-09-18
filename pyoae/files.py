"""Module providing functions for file handling."""

import json
import os
from pathlib import Path
from typing import Any

from pyoae import calib
from pyoae import get_logger
from pyoae.device.device_config import DeviceConfig
from pyoae import protocols

log = get_logger(__name__)

def load_json_file(file_path: str | Path) -> dict[str, Any]:
    """Loads the content of a json file."""

    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            output_dict = json.load(input_file)
    except FileNotFoundError:
        log.error('Could not load %s. File not found.', file_path)
        output_dict = {}

    return output_dict


def load_device_config(file_path: str) -> None:
    """Loads device configuration parameters from JSON."""
    config_data = load_json_file(file_path)
    if config_data:
        for key, entry in config_data.items():
            DeviceConfig.set(key, entry)
    else:
        log.error('Failed to load device configuration from %s', file_path)


def load_micro_calib(file_path: str | Path) -> calib.MicroCalibData | None:
    """Loads the microphone calibration data from JSON."""
    d = {}
    if file_path:
        d = load_json_file(file_path)
    else:
        log.error('Micro calibration %s not found.', file_path)
        return None
    micro_calib_data = calib.get_empty_micro_calib_data()

    if d:
        for key in micro_calib_data:
            if key in d:
                micro_calib_data[key] = d[key]

    return micro_calib_data


def load_output_calib(file_path: str) -> calib.SpeakerCalibData:
    """Loads the output calibration from a JSON."""
    d = {}
    if file_path:
        d = load_json_file(file_path)
    else:
        log.error('Output calibration %s not found.', file_path)
    out_calib_data = calib.get_empty_speaker_calib_data()

    if d:
        for key in out_calib_data:
            if key in d:
                out_calib_data[key] = d[key]

    return out_calib_data


def load_soae_protocol(
    file_path: str | None = None
) -> protocols.SoaeMsrmtParams:
    """Loads the SOAE measurement parameters or returns default."""
    msrmt_params = protocols.get_default_soae_msrmt_params()
    d = {}
    if file_path:
        d = load_json_file(file_path)
    else:
        log.error('SOAE protocol %s not found.', file_path)
        log.warning('Using default SOAE protocol instead.')

    if d:
        for key in msrmt_params:
            if key in d:
                msrmt_params[key] = d[key]

    return msrmt_params


def load_dpoae_protocol(
    file_path: str
) -> list[protocols.DpoaeMsrmtParams]:
    """Loads a DPOAE measurement protocol."""
    if file_path:
        d = load_json_file(file_path)
        # TODO: check content of protocol
        if 'msrmts' in d:
            return d['msrmts']
    else:
        log.error('Failed to load DPOAE protocol from %s', file_path)
    return []


def load_pulsed_dpoae_protocol(
    file_path: str
) -> list[protocols.PulseDpoaeMsrmtParams]:
    """Loads a Pulse-DPOAE measurement protocol."""
    if file_path:
        d = load_json_file(file_path)
        # TODO: verify content of protocol
        if 'msrmts' in d:
            return d['msrmts']
    else:
        log.error('Failed to load pulsed DPOAE protocol from %s', file_path)
    return []


def save_output_calibration(
    out_calib: calib.SpeakerCalibData
) -> None:
    """Saves output calibration results to JSON."""
    file_path = os.path.join(
        os.getcwd(),
        'measurements'
    )
    os.makedirs(file_path, exist_ok=True)
    file_name = out_calib['date'] + '_out_calib.json'
    file_path = os.path.join(file_path, file_name)

    try:
        with open(file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(out_calib, output_file, indent="\t")
        log.info('Output calibration saved to %s.', file_name)
    except (FileNotFoundError, TypeError, ValueError) as e:
        log.error('Error saving to %s - %s.', file_path, e)


def save_result_to_json(file_path: str | Path, data: dict) -> None:
    """Saves processed data to JSON file for further processing."""
    try:
        with open(file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(data, output_file, indent="\t")

    except (FileNotFoundError, TypeError, ValueError) as e:
        print(f'Error saving {file_path}.')
        print(e)
    print('Results saved to {file_path}.')
