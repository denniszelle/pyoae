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
from pyoae.device.device_config import DeviceConfig
from pyoae.soae import SoaeRecorder
import pyoae_logger

DEVICE_CONFIG_FILE = 'device_config.json'


logger = pyoae_logger.get_pyoae_logger('PyOAE SOAE Recorder')

def main(
    protocol: str = '',
    subject: str = '',
    ear: str = '',
    save: bool = False
) -> None:
    """Main function executing an SOAE measurement."""

    logger.info('SOAE recorder started with following options:')

    if protocol:
        logger.info('  Protocol: %s', protocol)
    if subject:
        logger.info('  Subject ID: %s', subject)
    if ear:
        logger.info('  Ear: %s', ear)
    if save:
        logger.info('Recording will be saved.')

    logger.info('Loading global configuration from %s.', DEVICE_CONFIG_FILE)
    files.load_device_config(DEVICE_CONFIG_FILE)
    logger.info('Device Configuration: %s', DeviceConfig())

    msrmt_params = files.load_soae_protocol(protocol)
    soae_recorder = SoaeRecorder(
        msrmt_params,
        subject=subject,
        ear=ear
    )
    soae_recorder.record()
    if save:
        soae_recorder.save_recording()


parser = argparse.ArgumentParser(description='PyOAE SOAE Recorder')
parser.add_argument(
    '--protocol',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to measurement protocol JSON file.'
)
parser.add_argument(
    '--subject',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify subject ID, e.g. S001.'
)
parser.add_argument(
    '--ear',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify the recording side, left/right, l/r.'
)

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
