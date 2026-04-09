"""Module to validate user input for the record script."""

from typing import Any, TypeVar

import numpy as np

from pyoae import get_logger
from pyoae.device.device_config import DeviceConfig

logger = get_logger(__name__)

T = TypeVar('T')


def validate_output_channels(output_channels: list[int]) -> bool:
    """Validate that the specified output channels exist in the device configuration.

    Checks each output channel against the device's output-input mapping
    defined in `DeviceConfig`. Logs an error if a channel is not valid.

    Args:
        output_channels: A list of integers representing output channel indices.

    Returns:
        True if all channels are valid, False if any channel is invalid.
    """
    for output_channel_i in output_channels:
        if not (
            output_channel_i
            in np.asarray(DeviceConfig.output_input_mapping)[:,0]
        ):

            logger.error(
                'Invalid channel selection. Channel %s has no mapping '
                'defined in the device_config file.',
                output_channel_i
            )
            return False
    return True


def validate_msrmt_params(msrmt_params: list[T] | T) -> list[T]:
    """Validate measurement parameters and ensure they are returned as a list.

    This function checks if the input measurement parameters are provided. If a single
    parameter is given, it is wrapped in a list. If an empty list is provided, an error
    is logged.

    Args:
        msrmt_params: A single measurement parameter or a list of parameters of type T.

    Returns:
        A list of measurement parameters. Returns an empty list if no parameters are provided.
    """
    if isinstance(msrmt_params, list):
        if len(msrmt_params) == 0:
            logger.error('No measurement parameters given for measurement.')
            return []
    else:
        msrmt_params = [msrmt_params]
    return msrmt_params


def validate_mic_tfs(
    micro_tfs: list[Any] | None,
    msrmt_params: list[Any]
) -> bool:
    """Validate that number of microphone TFs matches measurement parameters.

    Args:
        micro_tfs: A list of microphone transfer functions or `None` if not used.
        msrmt_params: A list of measurement parameters.

    Returns:
        True if `micro_tfs` is `None` or its length matches `msrmt_params`;
        otherwise False.
    """
    if micro_tfs is None:
        return True

    if len(micro_tfs) == len(msrmt_params):
        return True

    return False
