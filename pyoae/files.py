"""Module providing functions for file handling."""

import json

from typing import Any

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
