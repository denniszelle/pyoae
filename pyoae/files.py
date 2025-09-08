"""Module providing functions for file handling."""

import json
import os
from typing import Any

from pyoae import calib
from pyoae.device.device_config import DeviceConfig
from pyoae import protocols

def load_json_file(file_path: str) -> dict[str, Any]:
    """Loads the content of a json file."""

    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            output_dict = json.load(input_file)
    except FileNotFoundError:
        output_dict = {}

    return output_dict


def load_device_config(file_path: str) -> None:
    """Loads device configuration parameters from JSON."""
    config_data = load_json_file(file_path)
    if config_data:
        for key, entry in config_data.items():
            DeviceConfig.set(key, entry)
    # TODO: log errors if configuration was not found or was invalid


def load_micro_calib(file_path: str) -> calib.MicroCalibData:
    """Loads the microphone calibration data from JSON."""
    d = {}
    if file_path:
        d = load_json_file(file_path)
    micro_calib_data = calib.get_empty_micro_calib_data()

    if d:
        for key in micro_calib_data:
            if key in d:
                micro_calib_data[key] = d[key]

    return micro_calib_data


def load_soae_protocol(file_path: str | None = None) -> protocols.SoaeMsrmtParams:
    """Loads the SOAE measurement parameters or returns default."""
    msrmt_params = protocols.get_default_soae_msrmt_params()
    d = {}
    if file_path:
        d = load_json_file(file_path)

    if d:
        for key in msrmt_params:
            if key in d:
                msrmt_params[key] = d[key]

    return msrmt_params


def load_dpoae_protocol(file_path: str) -> list[protocols.DpoaeMsrmtParams]:
    """Loads a DPOAE measurement protocol."""
    # TODO: check content of protocol
    if file_path:
        d = load_json_file(file_path)
        if 'msrmts' in d:
            return d['msrmts']
    return []


def save_output_calibration(
    time_stamp: str,
    out_calib: calib.SpeakerCalibData
) -> None:
    """Saves output calibration results to JSON."""
    file_path = os.path.join(
        os.getcwd(),
        'measurements'
    )
    os.makedirs(file_path, exist_ok=True)
    file_name = time_stamp + 'out_calib.json'
    file_path = os.path.join(file_path, file_name)

    try:
        with open(file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(out_calib, output_file, indent="\t")
            #json.dump(out_calib, output_file, indent="\t", cls=ExportDataNumpyEncoder)

    except (FileNotFoundError, TypeError, ValueError) as e:
        print(f'Error saving {file_path}.')
        print(e)
    print('Output calibration saved to {file_name}.')
