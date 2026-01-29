"""Module to validate user input for the record script."""

from typing import TypeVar

import numpy as np

from pyoae import get_logger
from pyoae.device.device_config import DeviceConfig


logger = get_logger(__name__)


T = TypeVar('T')


def validate_output_channels(output_channels: list[int]) -> bool:
    """Validate the list of output channels"""
    for output_channel_i in output_channels:
        if not output_channel_i in np.asarray(DeviceConfig.output_input_mapping)[:,0]:
            logger.error(
                'Invalid channel selection. Channel %s has no mapping '
                'defined in the device_config file.',
                output_channel_i
            )
            return False
    return True

def validate_msrmt_params(msrmt_params: list[T] | T) -> list[T]:
    """Validate measurement parameters"""
    if isinstance(msrmt_params, list):
        if len(msrmt_params) == 0:
            logger.error('No measurement parameters given for measurement.')
            return []
    else:
        msrmt_params = [msrmt_params]
    return msrmt_params
