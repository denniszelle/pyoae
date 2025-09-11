"""Script to record pulsed DPOAEs.

This script performs a real-time acquisition of pulsed distortion-product
otoacoustic emissions (DPOAEs) using two pulsed primary tones (f1 and f2).

This script is intended as an example and a starting point for continuous
DPOAE measurements.


Run the following command from the project root directory to start:

    python3 -m record_pulse_dpoae --mic '2ROTIU6H_3C9CESK1W6.json'
      --protocol 'protocols/f2sp_ieb_ae.json' --calib '250910-180811'
      --subject 'AE' --ear 'right'  --save

Note:
    Sound device IDs or names should be known beforehand and can be
      obtained using the `display_devices` script.

    Users should modify parameters defined in the `device_config.json`
      to match their specific hardware and experimental setup.
"""

import argparse
import os

from pyoae import files
from pyoae.calib import MicroTransferFunction, OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.pdpoae import PulseDpoaeRecorder


DEVICE_CONFIG_FILE = 'device_config.json'


def main(
    protocol: str = '',
    mic: str = '',
    calib: str = '',
    subject: str = '',
    ear: str = '',
    save: bool = False
) -> None:
    """Main function executing a pulsed DPOAE measurement."""

    print('Pulse DPOAE recorder started with following options:')
    print(f'  Protocol: {protocol}')
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

    if calib:
        print('Loading output calibration.')
        out_file_name = calib + '_out_calib.json'
        out_file_path = os.path.join(os.getcwd(), 'measurements', out_file_name)
        speaker_calib_data = files.load_output_calib(out_file_path)
        output_calib_fun = OutputCalibration(speaker_calib_data)
    else:
        output_calib_fun = None

    protocol_path = os.path.join(os.getcwd(), protocol)
    dpoae_protocol = files.load_pulsed_dpoae_protocol(protocol_path)
    for msrmt_params in dpoae_protocol:
        dpoae_recorder = PulseDpoaeRecorder(
            msrmt_params,
            mic_trans_fun,
            out_trans_fun=output_calib_fun,
            subject=subject,
            ear=ear
        )
        dpoae_recorder.record()
        if save:
            dpoae_recorder.save_recording()


parser = argparse.ArgumentParser(description='PyOAE Pulse-DPOAE Recorder')
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
