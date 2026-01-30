"""Script to perform an absolute calibration of the input channel.

This script demonstrates how to calibrate the sensitivity of the input
system. It uses a synchronized playback and recording setup. A live
time-domain and frequency-domain plot is shown during the measurement.
After the recording, the script shows the offline results with the
maximum peak pressure that corresponds the a digital input amplitude
of 1.0 re FS.

Run the following command from the project root directory to start:
    calibrate_sensitivity

Note:
    Sound device IDs or names should be known beforehand and can be obtained
      using the display_devices script.

    Users should modify parameters defined in the `device_config.json` file
      to match their specific hardware and experimental setup.
"""

import argparse

from pyoae import files
from pyoae import protocols
from pyoae import pyoae_logger
from pyoae.device.device_config import DeviceConfig
from pyoae.abs_calib import AbsCalibRecorder


DEVICE_CONFIG_FILE = 'device_config.json'


logger = pyoae_logger.get_pyoae_logger(
    'PyOAE Input Sensitivity Calibration'
)

def main(
    save: bool = False,
    input_ch: int | None = None,
) -> None:
    """Main function executing a sensitivity measurement."""

    logger.info('Sensitivity calibration.')
    if save:
        logger.info('Recording will be saved.')
    if input_ch is None:
        input_ch = 0

    logger.info('Loading global configuration from %s.', DEVICE_CONFIG_FILE)
    files.load_device_config(DEVICE_CONFIG_FILE)
    logger.info('Device Configuration: %s', DeviceConfig())

    msrmt_params = protocols.get_default_soae_msrmt_params()
    recorder = AbsCalibRecorder(msrmt_params, [input_ch])
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
parser.add_argument(
    '--mic',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to microphone calibration JSON file.'
)


def run_cli() -> None:
    """Run main with console arguments"""
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)


if __name__ == "__main__":
    # Entry point for console module execution
    run_cli()
