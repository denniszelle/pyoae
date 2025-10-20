"""Script to show calibration data of output channels.

This script shows the results of a multi-tone calibration of output channels
with respect to a given input calibration.


Run the following command from the project root directory to start:

    python3 -m show_output_calib --file '251015-190856_out_calib.json'

Command-line arguments:
    --file: path to output-calibration file to be shown, e.g., '251015-190856_out_calib.json'

"""

import argparse

from pyoae import files
from pyoae.calib import OutputCalibration
from pyoae import calibrator
import pyoae_logger


logger = pyoae_logger.get_pyoae_logger('PyOAE Output Calibrator')

def main(file: str = '') -> None:
    """Main function executing a multi-tone measurement."""

    logger.info('Display of multi-tone output calibration results.')

    if file:
        logger.info('Loading output calibration from  %s.', file)
        speaker_calib_data = files.load_output_calib(file)
        if speaker_calib_data['frequencies']:
            output_calib_fun = OutputCalibration(speaker_calib_data)
        else:
            output_calib_fun = None
    else:
        output_calib_fun = None

    if output_calib_fun is None:
        logger.error('Failed to load output calibration.')
        return
    calibrator.plot_result_file(output_calib_fun)


parser = argparse.ArgumentParser(description='PyOAE Multi-Tone Calibration')
parser.add_argument(
    '--file',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify output calibration file.'
)


if __name__ == "__main__":
    # Entry point for script execution
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
