"""Script to calibrate output channels.

This script performs a multi-tone calibration of output channels
with respect to a given input calibration. Typically, it is used
to calibrate two speakers of a DPOAE ear probe.


Run the following command from the project root directory to start:

    python3 -m record_output_calib --mic 'mic/mic_calib.json' --save


Note:
    Sound device IDs or names should be known beforehand and can be obtained
      using the display_devices.

    Users should modify parameters defined in the `device_config.json`
      to match their specific hardware and experimental setup.
"""

import argparse

from pyoae import files
from pyoae import input_validation
from pyoae import protocols
from pyoae.calib import MicroTransferFunction
from pyoae.device.device_config import DeviceConfig
from pyoae.calibrator import OutputCalibRecorder
import pyoae.pyoae_logger as pyoae_logger

DEVICE_CONFIG_FILE = 'device_config.json'

logger = pyoae_logger.get_pyoae_logger('PyOAE Output Calibrator')

def main(
    mic: list[str] | None = None,
    save: bool = False,
    protocol: str = '',
    channels: list[int] | None = None,
) -> None:
    """Main function executing a multi-tone measurement."""

    logger.info('Multi-tone output calibrator started.')
    if save:
        logger.info('Calibration will be saved.')

    if channels is None:
        logger.info('Setting default output channels 0 and 1')
        channels = [0, 1]

    logger.info('Loading global configuration from %s.', DEVICE_CONFIG_FILE)
    files.load_device_config(DEVICE_CONFIG_FILE)
    logger.info('Device Configuration: %s', DeviceConfig())

    if not input_validation.validate_output_channels(channels):
        return

    if mic:
        logger.info('Loading microphone calibration from %s.', mic)
        mic_trans_fun = []
        for mic_i in mic:
            mic_calib_data = files.load_micro_calib(mic_i)
            if mic_calib_data is None:
                logger.error('Stopping: Failed to load microphone calibration.')
                return
            mic_trans_fun.append(
                MicroTransferFunction(
                    mic_calib_data['abs_calibration'],
                    mic_calib_data['transfer_function']
                )
            )
    else:
        mic_trans_fun = None

    if protocol:
        logger.info('Loading speaker calibration protocol from %s.', protocol)
        prtcl_data = files.load_json_file(protocol)
        msrmt_params = protocols.get_custom_calib_msrmt_params(prtcl_data)
        if msrmt_params is None:
            logger.error('Stopping: Output calibration protocol is invalid.')
            return
    else:
        msrmt_params = protocols.get_default_calib_msrmt_params()

    calib_recorder = OutputCalibRecorder(
        msrmt_params,
        output_channels=channels,
        mic_trans_fun=mic_trans_fun
    )
    calib_recorder.record()
    if save:
        calib_recorder.save_recording()
        if calib_recorder.results is not None:
            calib_id = calib_recorder.results["date"]
            logger.info('Calibration saved with time stamp: %s.', calib_id)


parser = argparse.ArgumentParser(description='PyOAE Multi-Tone Calibration')
parser.add_argument(
    '--channels',
    nargs='+',
    default=argparse.SUPPRESS,
    type=int
)
parser.add_argument(
    '--mic',
    nargs='+',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to microphone calibration JSON file.'
)
parser.add_argument(
    '--protocol',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to output calibration protocol JSON file.'
)
parser.add_argument(
    '--save',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Save measurement results and data to files.'
)


def run_cli():
    """Run main with console arguments"""
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)


if __name__ == "__main__":
    # Entry point for console module execution
    run_cli()
