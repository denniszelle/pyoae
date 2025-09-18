"""Example script to record SOAEs (Spontaneous Otoacoustic Emissions).

This script demonstrates how to acquire and visualize SOAE data in real-time.
It uses a synchronized playback and recording setup. A live time-domain
and frequency-domain plot is shown during the measurement.

Key steps performed by this script:
- Configures signal generators and buffer parameters
- Sets up real-time plots for signal and spectrum display
- Runs a synchronized measurement using input and output sound devices
- Saves the recorded signal and sampling rate to a .npz file

This script is intended to be used as an example and entry point for
SOAE measurements.

Run the following command from the project root directory to start:
    python -m record_soae

Note:
    Sound device IDs or names should be known beforehand and can be obtained
      using the display_devices script.

    Users should modify parameters defined in the `device_config` module
      to match their specific hardware and experimental setup.
"""

import argparse

from pyoae import files
from pyoae import protocols
from pyoae.device.device_config import DeviceConfig
from pyoae.abs_calib import AbsCalibRecorder
import pyoae_logger

DEVICE_CONFIG_FILE = 'device_config.json'


logger = pyoae_logger.get_pyoae_logger(
    'PyOAE Input Sensitivity Calibration'
)

def main(
    save: bool = False
) -> None:
    """Main function executing a sensitivity measurement."""

    logger.info('Sensitivity calibration.')
    if save:
        logger.info('Recording will be saved.')

    logger.info('Loading global configuration from %s.', DEVICE_CONFIG_FILE)
    files.load_device_config(DEVICE_CONFIG_FILE)
    logger.info('Device Configuration: %s', DeviceConfig())

    msrmt_params = protocols.get_default_soae_msrmt_params()
    recorder = AbsCalibRecorder(msrmt_params)
    recorder.record()
    if save:
        recorder.save_recording()


parser = argparse.ArgumentParser(description='PyOAE Sensitivity Calibration')
parser.add_argument(
    '--save',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Save measurement results and data to files.'
)


if __name__ == "__main__":
    # Entry point for script execution
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
