"""Script to record continuous DPOAEs.

This script performs a real-time acquisition of continuous distortion-product
otoacoustic emissions (DPOAEs) using two primary tones (f1 and f2).
The recorded microphone signal is analyzed in real time, and live plots of
the time signal and spectrum are displayed.


Run the following command from the project root directory to start:
    python -m record_dpoae

Note:
    Sound device IDs or names should be known beforehand and can be
      obtained using the display_devices script.

    Users should modify parameters defined in the `device_config.json`
      to match their specific hardware and experimental setup.
"""

import argparse
import os

from pyoae import files
from pyoae.calib import MicroTransferFunction, OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.cdpoae import DpoaeRecorder
import pyoae_logger


DEVICE_CONFIG_FILE = 'device_config.json'


logger = pyoae_logger.get_pyoae_logger('PyOAE cDPOAE Recorder')


def main(
    protocol: str = '',
    mic: str = '',
    calib: str = '',
    subject: str = '',
    ear: str = '',
    save: bool = False
) -> None:
    """Main function executing a DPOAE measurement."""

    logger.info('DPOAE recorder started with following options:')
    if protocol:
        logger.info('  Protocol: %s', protocol)
    else:
        logger.error('Please specify a measurement protocol.')
        return
    if subject:
        logger.info('  Subject ID: %s', subject)
    if ear:
        logger.info('  Ear: %s', ear)
    if save:
        logger.info('Recording will be saved.')

    logger.info('Loading global configuration from %s.', DEVICE_CONFIG_FILE)
    files.load_device_config(DEVICE_CONFIG_FILE)
    logger.info('Device Configuration: %s', DeviceConfig())

    if mic:
        logger.info('Loading microphone calibration from %s.', mic)
        mic_calib_data = files.load_micro_calib(mic)
        if mic_calib_data is None:
            logger.error('Stopping: Failed to load microphone calibration.')
            return
        mic_trans_fun = MicroTransferFunction(
            mic_calib_data['abs_calibration'],
            mic_calib_data['transfer_function']
        )
    else:
        mic_trans_fun = None

    if calib:
        logger.info('Loading output calibration with id %s.', calib)
        out_file_name = calib + '_out_calib.json'
        out_file_path = os.path.join(os.getcwd(), 'measurements', out_file_name)
        speaker_calib_data = files.load_output_calib(out_file_path)
        if speaker_calib_data['frequencies']:
            output_calib_fun = OutputCalibration(speaker_calib_data)
        else:
            output_calib_fun = None
    else:
        output_calib_fun = None

    protocol_path = os.path.join(os.getcwd(), protocol)
    dpoae_protocol = files.load_dpoae_protocol(protocol_path)
    for msrmt_params in dpoae_protocol:
        dpoae_recorder = DpoaeRecorder(
            msrmt_params,
            mic_trans_fun,
            out_trans_fun=output_calib_fun,
            subject=subject,
            ear=ear
        )
        dpoae_recorder.record()
        if save:
            dpoae_recorder.save_recording()


parser = argparse.ArgumentParser(description='PyOAE DPOAE Recorder')
parser.add_argument(
    '--protocol',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to measurement protocol JSON file.'
)
parser.add_argument(
    '--mic',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify path to microphone calibration JSON file.'
)
parser.add_argument(
    '--calib',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify output calibration time stamp.'
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
