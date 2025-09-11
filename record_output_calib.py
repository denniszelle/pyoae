"""Script to calibrate output channels.

This script performs a multi-tone calibration of output channels
with respect to a given input calibration. Typically, it is used
to calibrate two speakers of a DPOAE ear probe.


Run the following command from the project root directory to start:

    python3 -m record_output_calib --mic '2ROTIU6H_3C9CESK1W6.json' --save


Note:
    Sound device IDs or names should be known beforehand and can be obtained
      using the display_devices.

    Users should modify parameters defined in the `device_config.json`
      to match their specific hardware and experimental setup.
"""

import argparse

from pyoae import files
from pyoae import protocols
from pyoae.calib import MicroTransferFunction
from pyoae.device.device_config import DeviceConfig
from pyoae.calibrator import OutputCalibRecorder


DEVICE_CONFIG_FILE = 'device_config.json'


def main(
    mic: str = '',
    subject: str = '',
    ear: str = '',
    save: bool = False
) -> None:
    """Main function executing a multi-tone measurement."""

    print('Multi-tone output calibrator started with following options:')
    print(f'  Subject ID: {subject} - ear: {ear}')
    if save:
        print('Recording will be saved in files.')

    print('Loading configuration.')
    files.load_device_config(DEVICE_CONFIG_FILE)
    print(DeviceConfig())

    if mic:
        print('Loading microphone calibration.')
        mic_calib_data = files.load_micro_calib(mic)
        mic_trans_fun = MicroTransferFunction(
            mic_calib_data['abs_calibration'],
            mic_calib_data['transfer_function']
        )

    else:
        mic_trans_fun = None

    #dpoae_protocol = files.load_dpoae_protocol(protocol)
    msrmt_params = protocols.get_default_calib_msrmt_params()
    calib_recorder = OutputCalibRecorder(
        msrmt_params,
        mic_trans_fun=mic_trans_fun
    )
    calib_recorder.record()
    if save:
        calib_recorder.save_recording()
        if calib_recorder.results is not None:
            calib_id = calib_recorder.results["date"]
            print(f'Calibration saved with time stamp: {calib_id}.')


parser = argparse.ArgumentParser(description='PyOAE Multi-Tone Calibration')
parser.add_argument(
    '--mic',
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
