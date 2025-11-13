"""Script to show continuous DPOAE data.

This script shows the results of a conventional DPOAE
recording using continuously presented primary tones.


Run the following command from the project root directory to start:

    python3 -m show_dpoae --file 'cdpoae_msrmt_250915-172224.npz'

Command-line arguments:
    --file: path to result file to be shown, e.g., 'cdpoae_msrmt_250915-172224.npz'

"""

import argparse

from pyoae import files
from pyoae.dsp.processing import ContDpoaeResult
import pyoae_logger


logger = pyoae_logger.get_pyoae_logger('PyOAE DPOAE Results')

def main(file: str = '') -> None:
    """Main function visualizing a DPOAE spectrum."""

    logger.info('Display of results from continuous DPOAE recording.')

    cdpoae_result = None
    if file:
        logger.info('Loading DPOAE results from  %s.', file)
        cont_recording = files.load_cdpoae_recording(file)
        if cont_recording is not None:
            cdpoae_result = ContDpoaeResult(cont_recording)

    if cdpoae_result is None:
        logger.error('Failed to load continuous DPOAE result.')
        return

    cdpoae_result.plot()


parser = argparse.ArgumentParser(description='PyOAE CDPOAE Result')
parser.add_argument(
    '--file',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify CDPOAE recording file.'
)


if __name__ == "__main__":
    # Entry point for script execution
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
