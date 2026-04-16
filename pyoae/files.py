"""Module providing functions for file handling."""

import csv
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

from pyoae import calib_storage
from pyoae.calib_storage import(
    MicroCalibData,
    SpeakerCalibData
)
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
    """Load the contents of a JSON file into a Python dictionary.

    If the file does not exist, an empty dictionary is returned and an error
    is logged.

    Args:
        file_path: Path to the JSON file to load. Can be a string or a
            `Path` object.

    Returns:
        A dictionary containing the JSON data. Returns an empty dictionary
        if the file is not found.

    Raises:
        None. File-not-found errors are handled internally by logging and
        returning an empty dictionary.
    """

    try:
        with open(file_path, 'r', encoding='utf-8') as input_file:
            output_dict = json.load(input_file)
    except FileNotFoundError:
        log.error('Could not load %s. File not found.', file_path)
        output_dict = {}

    return output_dict


def load_device_config(file_path: str) -> None:
    """Load device configuration parameters from JSON file into `DeviceConfig`.

    The JSON file should contain key-value pairs corresponding to device
    configuration settings. Each entry is set in the `DeviceConfig` using
    `DeviceConfig.set(key, value)`.

    Args:
        file_path: Path to the JSON file containing device configuration.

    Returns:
        None. The device configuration is updated in-place.

    Raises:
        None. File-not-found or empty JSON errors are logged internally.
    """
    config_data = load_json_file(file_path)
    if config_data:
        for key, entry in config_data.items():
            DeviceConfig.set(key, entry)
    else:
        log.error('Failed to load device configuration from %s', file_path)


def load_micro_calib(
    file_path: str | Path
) -> MicroCalibData | None:
    """Load microphone calibration data from a JSON file.

    Reads a JSON file containing microphone calibration information and
    returns a `MicroCalibData` dictionary with all expected fields. Missing
    fields are filled with default values.

    Args:
        file_path: Path to the JSON file containing the microphone calibration
            data. Can be a string or a `Path` object.

    Returns:
        A `MicroCalibData` dictionary with calibration data if the file was
        successfully loaded; `None` if the file does not exist or cannot be
        loaded.

    Raises:
        None. File-not-found errors are logged internally and result in `None`.
    """
    d = {}
    if file_path:
        d = load_json_file(file_path)
    if not d:
        log.error('Micro calibration %s not found.', file_path)
        return None
    micro_calib_data = calib_storage.get_empty_micro_calib_data()

    if d:
        for key in micro_calib_data:
            if key in d:
                micro_calib_data[key] = d[key]

    return micro_calib_data


def load_csv_output_calib(file_path: str) -> protocols.CalibMsrmtDef | None:
    """Load output calibration measurement definition from a CSV file.

    The CSV file is expected to contain two sections:
      1. Metadata at the top of the file (key,value pairs) until the first empty line.
      2. Data section with columns: 'frequencies', 'phases', 'amplitudes', 'cluster_idx'.

    Args:
        file_path: Path to the CSV file containing the output calibration data.

    Returns:
        A `CalibMsrmtDef` dictionary containing:
            - 'block_duration': duration of each measurement block (float)
            - 'num_averaging_blocks': number of averaging blocks (int)
            - 'frequencies': array of frequencies (np.ndarray[float])
            - 'phases': array of phases (np.ndarray[float])
            - 'amplitudes': array of amplitudes (np.ndarray[float])
            - 'cluster_idc': array of cluster indices (np.ndarray[int])
        Returns `None` if the file cannot be read or is empty.

    Raises:
        None. File reading errors will propagate naturally (e.g., FileNotFoundError)
        or may result in exceptions from malformed CSV data.
    """

    meta = {}
    data_rows = []

    with open(file_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        # --- Read metadata ---
        for row in reader:
            if not row:  # empty line -> end of metadata
                break
            key, value = row
            meta[key] = value

        # --- Read header ---
        _ = next(reader)
        # ['frequencies', 'phases', 'amplitudes', 'cluster_idx']
        # TODO: Check for valid headers.

        # --- Read data ---
        for row in reader:
            if row:  # skip possible empty lines
                data_rows.append(row)

    # Convert to numpy array with correct types
    data = np.array(data_rows)

    # Convert metadata to proper types
    meta['block_duration'] = float(meta['block_duration'])
    meta['num_averaging_blocks'] = int(meta['num_averaging_blocks'])

    msrmt_params: protocols.CalibMsrmtDef = {
        'block_duration': float(meta['block_duration']),
        'num_averaging_blocks': int(meta['num_averaging_blocks']),
        'frequencies': data[:, 0].astype(float),
        'phases': data[:, 1].astype(float),
        'amplitudes': data[:, 2].astype(float),
        'cluster_idc': data[:, 3].astype(int)
    }
    return msrmt_params


def load_output_calib_protocol(
    file_path: str
) -> (protocols.CalibMsrmtParams | protocols.CalibMsrmtDef | None):
    """Load an output calibration protocol from a CSV or JSON file.

    This function determines the file type based on the extension:
      - `.csv`: parsed as a measurement definition using `load_csv_output_calib`.
      - `.json`: parsed as custom calibration parameters using
        `protocols.get_custom_calib_msrmt_params`.

    Args:
        file_path: Path to the calibration protocol file.

    Returns:
        Either:
        - `CalibMsrmtDef` if a CSV file is loaded,
        - `CalibMsrmtParams` if a JSON file is loaded,
        - `None` if the file does not exist, is not a file, or has an invalid
          extension.
    """
    path = Path(file_path)

    if not path.exists():
        log.error('Could not find calibration protocol file.')
        return None

    if not path.is_file():
        log.error('Given path is not a file.')
        return None

    suffix = path.suffix.lower()

    if suffix == '.csv':
        return load_csv_output_calib(file_path)

    elif suffix == '.json':
        prtcl_data = load_json_file(file_path)
        return protocols.get_custom_calib_msrmt_params(prtcl_data)
    else:
        log.error(
            'Invalid file format. Only .csv or .json are valid as '
            'output calib protocol'
        )
        return None


def load_output_calib(file_path: str) -> SpeakerCalibData:
    """Load speaker/output calibration data from a JSON file.

    The function attempts to read the JSON file at the given path and populate
    a `SpeakerCalibData` container. If the file cannot be found or is empty,
    an empty calibration container is returned and an error is logged.

    Args:
        file_path: Path to the JSON file containing the speaker calibration data.

    Returns:
        A `SpeakerCalibData` dictionary populated with the loaded calibration
        values. If the file is missing or empty, returns a default empty container.
    """
    d = {}
    if file_path:
        d = load_json_file(file_path)

    if not d:
        log.error('Output calibration %s not found.', file_path)
    out_calib_data = calib_storage.get_empty_speaker_calib_data()

    if d:
        for key in out_calib_data:
            if key in d:
                out_calib_data[key] = d[key]

    return out_calib_data


def load_soae_protocol(file_path: str | None = None) -> protocols.MsrmtParams:
    """Load SOAE measurement parameters from a JSON file or return defaults.

    Args:
        file_path: Optional path to a JSON file containing SOAE measurement
            parameters. If None or the file cannot be loaded, default parameters
            are used.

    Returns:
        A `protocols.MsrmtParams` dictionary containing the loaded or default
        SOAE measurement parameters.
    """
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
    """Load a DPOAE measurement protocol from a JSON file.

    Args:
        file_path: Path to the JSON file containing the DPOAE protocol.

    Returns:
        A list of `protocols.DpoaeMsrmtParams` dictionaries representing the
        measurement protocol. If multiple blocks are present, a nested list is
        returned. Returns an empty list if the file cannot be loaded or if the
        `'msrmts'` key is missing.
    """
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
    """Load a pulsed DPOAE measurement protocol from a JSON file.

    This function reads a Pulse Distortion Product Otoacoustic Emission
    (Pulse-DPOAE) measurement protocol from the specified JSON file. The file
    is expected to contain a top-level key `'msrmts'` with a list of
    measurement parameter dictionaries.

    Args:
        file_path: Path to the JSON file containing the Pulse-DPOAE protocol.

    Returns:
        A list of `protocols.PulseDpoaeMsrmtParams` dictionaries representing the
        measurement protocol. Returns an empty list if the file cannot be loaded or
        if the `'msrmts'` key is missing.
    """
    if file_path:
        d = load_json_file(file_path)
        # TODO: verify content of protocol
        if 'msrmts' in d:
            return d['msrmts']
    else:
        log.error('Failed to load pulsed DPOAE protocol from %s', file_path)
    return []


def allows_none(tp) -> bool:
    """Check whether a type annotation allows `None`.

    This function examines a type hint and returns `True` if `None` is
    allowed, such as in `Optional[...]` or a union type containing `None`.

    Args:
        tp: A type annotation to inspect. Can be a `Union`, `Optional`, or
            PEP 604 union (e.g., `int | None`).

    Returns:
        True if `None` is an allowed type in the annotation; False otherwise.
    """
    origin = get_origin(tp)
    # handle both classic Union and PEP 604 |
    if origin in (Union, types.UnionType):
        return type(None) in get_args(tp)
    return False


def load_typed_dict(schema: Type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Load a TypedDict from a dictionary and cast numeric fields automatically.

    This function takes a dictionary `data` and a TypedDict `schema` class,
    and returns a new dictionary where fields defined in the TypedDict are
    cast to the correct type. It automatically converts numeric fields
    (`int` and `float`) to their expected Python types and handles optional
    fields (fields that allow `None`).

    Args:
        schema: A TypedDict class defining the expected fields and types.
        data: A dictionary containing the data to load into the TypedDict.

    Returns:
        A dictionary with keys matching the TypedDict fields and values
        cast to the appropriate types. Optional fields not present in `data`
        are set to `None`.
    """
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
    """Load a continuous DPOAE recording from a binary `.npz` or `.npy` file.

    Args:
        file_path: Path to the binary file containing the continuous DPOAE
            recording.

    Returns:
        A dictionary containing the recording data:

        - `'recording'`: The main measurement data, loaded and cast to
          `DpoaeMsrmtData`.
        - `'average'`: Optional averaged measurement data if present in the
            file, otherwise `None`.
        - `'spectrum'`: Optional spectrum data if present in the file,
        otherwise `None`.

        Returns `None` if the file could not be found or loaded.
    """
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

    return {'recording': recording, 'average': average, 'spectrum': spectrum}


def load_pdpoae_recording(file_path: str | Path) -> PulseDpoaeRecording | None:
    """Load a pulsed DPOAE recording from a binary `.npz` or `.npy` file.

    Args:
        file_path: Path to the binary file containing the pulsed DPOAE recording.

    Returns:
        A dictionary containing the recording data:

        - `'recording'`: The main measurement data, loaded and cast to
          `DpoaeMsrmtData`.
        - `'average'`: Optional raw average measurement data if present in the
          file, otherwise `None`.
        - `'signal'`: Optional precomputed average for backwards compatibility,
          otherwise `None`.

        Returns `None` if the file could not be found or loaded.
    """
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

    return {'recording': recording, 'average': raw_avg, 'signal': avg}


def save_output_calibration(out_calib: SpeakerCalibData) -> None:
    """Save output calibration data to a JSON file.

    The calibration is saved under the `measurements` folder in the current
    working directory. The filename is constructed from the calibration date
    in the format `<date>_out_calib.json`.

    Args:
        out_calib: A `SpeakerCalibData` dictionary containing the output
            calibration data to be saved. Must include a `'date'` field.

    Returns:
        None
    """
    file_path = os.path.join(os.getcwd(), 'measurements')
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
    """Save processed data to a JSON file for further use.

    The data dictionary is serialized to JSON and written to the specified
    file path. Creates or overwrites the file if it already exists.

    Args:
        file_path: Path (str or Path) where the JSON file should be saved.
        data: Dictionary containing the data to save. Must be JSON-serializable.

    Returns:
        None
    """
    try:
        with open(file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(data, output_file, indent="\t")

    except (FileNotFoundError, TypeError, ValueError) as e:
        log.error(f'Error saving {file_path}: {e}.')
        return
    log.info(f'Results saved to {file_path}.')
