"""Script to record SOAEs (Spontaneous Otoacoustic Emissions).

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

    Users should modify parameters defined in the `device_config.json` file
      to match their specific hardware and experimental setup.
"""

import argparse

from pyoae import files
from pyoae.calib import MicroTransferFunction
from pyoae.device.device_config import DeviceConfig
from pyoae.soae import SoaeRecorder
import pyoae.pyoae_logger as pyoae_logger

DEVICE_CONFIG_FILE = 'device_config.json'


logger = pyoae_logger.get_pyoae_logger('PyOAE SOAE Recorder')

def main(
    protocol: str = '',
    mic: list[str] | None = None,
    input_ch: list[int] | None = None,
    subject: str = '',
    ear: list[str] | None = None,
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
    if not input_ch:
        input_ch = [0]
    logger.info('Use input channels: %s', input_ch)

    logger.info('Loading global configuration from %s.', DEVICE_CONFIG_FILE)
    files.load_device_config(DEVICE_CONFIG_FILE)
    logger.info('Device Configuration: %s', DeviceConfig())

    if mic:
        logger.info('Loading microphone calibration from %s.', mic)
        mic_trans_fun = []
        for mic_i in mic:
            mic_calib_data = files.load_micro_calib(mic_i)
            if mic_calib_data is None:
                logger.error('Stopping: Failed to load microphone calibration.')
                return
            mic_trans_fun.append(MicroTransferFunction(
                    mic_calib_data['abs_calibration'],
                    mic_calib_data['transfer_function']
                )
            )
    else:
        mic_trans_fun = None

    msrmt_params = files.load_soae_protocol(protocol)
    soae_recorder = SoaeRecorder(
        msrmt_params,
        input_ch,
        mic_trans_fun,
        subject=subject,
        ear=ear
    )
    soae_recorder.record()
    if save:
        soae_recorder.save_recording()


parser = argparse.ArgumentParser(description='PyOAE SOAE Recorder')
parser.add_argument(
    '--input_ch',
    nargs='+',
    default=argparse.SUPPRESS,
    type=int
)
parser.add_argument(
    '--protocol',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to measurement protocol JSON file.'
)
parser.add_argument(
    '--mic',
    nargs='+',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to microphone calibration JSON file.'
)
parser.add_argument(
    '--subject',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify subject ID, e.g. S001.'
)
parser.add_argument(
    '--ear',
    nargs='+',
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


def run_cli() -> None:
    """Run main with console arguments"""
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)


if __name__ == "__main__":
    # Entry point for console module execution
    run_cli()
