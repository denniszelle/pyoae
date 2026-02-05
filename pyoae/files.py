"""Module providing functions for file handling."""

import json
import os
from pathlib import Path
from typing import (
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    Union,
    Type
)
import types
import numpy as np

from pyoae import calib
from pyoae import get_logger
from pyoae import protocols
from pyoae.device.device_config import DeviceConfig
from pyoae.dsp.containers import (
    ContDpoaeRecording,
    DpoaeMsrmtData,
    PulseDpoaeRecording
)


log = get_logger(__name__)


def find_npz_files(directory: str, prefix: str = '') -> list[Path]:
    """Returns all .npz-files in `directory` starting with `prefix`.

    Args:
        directory: str - Path to the directory to search for
          measurement files.
        prefix: str - Optional prefix to filter measurement files,
          e.g., 'cdpoae', 'pdpoae'.

    Returns:
        list[Path]: sorted list of matching file paths.
    """
    dir_path = Path(directory)
    return sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.name.startswith(prefix) and p.suffix == '.npz'
    )


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
    if not d:
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

    if not d:
        log.error('Output calibration %s not found.', file_path)
    out_calib_data = calib.get_empty_speaker_calib_data()

    if d:
        for key in out_calib_data:
            if key in d:
                out_calib_data[key] = d[key]

    return out_calib_data


def load_soae_protocol(
    file_path: str | None = None
) -> protocols.MsrmtParams:
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
) -> list[protocols.DpoaeMsrmtParams] | list[list[protocols.DpoaeMsrmtParams]]:
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


def allows_none(tp) -> bool:
    """Check whether a type field allows None."""
    origin = get_origin(tp)
    # handle both classic Union and PEP 604 |
    if origin in (Union, types.UnionType):
        return type(None) in get_args(tp)
    return False


def load_typed_dict(schema: Type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Load a TypedDict and automatically cast int/float fields."""
    result: dict[str, Any] = {}
    hints = get_type_hints(schema)

    for field, field_type in hints.items():
        if field in data:
            value = data[field]
            if value is not None:
                origin = get_origin(field_type)
                # Cast int and floats to python values
                if origin is None:
                    if field_type is int:
                        value = int(value)
                    elif field_type is float:
                        value = float(value)
            result[field] = value

        # Check whether None is allowed for this type
        elif allows_none(field_type):
            result[field] = None
        else:
            raise KeyError(f"Missing required field: {field}")

    return result


def load_cdpoae_recording(file_path: str | Path) -> ContDpoaeRecording | None:
    """Loads measurement data of a cont. DPOAE recording from binary file."""
    try:
        data = np.load(file_path)
    except FileNotFoundError as e:
        print(e)
        return None

    recording = cast(
        DpoaeMsrmtData,
        load_typed_dict(DpoaeMsrmtData, data),
    )

    if 'average' in data:
        average = data['average']
    else:
        average = None

    if 'spectrum' in data:
        spectrum = data['spectrum']
    else:
        spectrum = None

    return {
        'recording': recording,
        'average': average,
        'spectrum': spectrum
    }


def load_pdpoae_recording(file_path: str | Path) -> PulseDpoaeRecording | None:
    """Loads measurement data of a pulsed DPOAE recording from binary file."""
    try:
        data = np.load(file_path)
    except FileNotFoundError as e:
        print(e)
        return None

    recording = cast(
        DpoaeMsrmtData,
        load_typed_dict(DpoaeMsrmtData, data),
    )

    if 'raw_average' in data:
        raw_avg = data['raw_average']
    else:
        raw_avg = None

    if 'average' in data:
        # for backwards compatibility
        avg = data['average']
    else:
        avg = None

    return {
        'recording': recording,
        'average': raw_avg,
        'signal': avg
    }


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
