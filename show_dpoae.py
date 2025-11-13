"""Script to show continuous DPOAE data.

This script shows the results of a conventional DPOAE
recording using continuously presented primary tones.


Run the following command from the project root directory to
plot results from a single measurement:

    python3 -m show_dpoae --file 'cdpoae_msrmt_251021-141156_1500_60.npz'

Run the following command from the project root directory to
plot all continuous DPOAE recordings in a specified directory:

    python3 -m show_dpoae --d 'measurements'

Command-line arguments:
    --file: path to result file to be shown,
      e.g., 'cdpoae_msrmt_250915-172224.npz'
    --d: path to directory with multiple result files to be shown

"""

import argparse

from matplotlib import pyplot as plt

from pyoae import files
from pyoae.dsp.processing import ContDpoaeResult
import pyoae_logger


logger = pyoae_logger.get_pyoae_logger('PyOAE DPOAE Results')

def main(file: str = '', d: str = '', a: bool = False) -> None:
    """Main function visualizing a DPOAE spectrum."""

    logger.info('Display of results from continuous DPOAE recording.')

    cdpoae_result = None

    if d:
        logger.info('Loading DPOAE files from %s.', d)
        # plot results from all cDPOAE measurement files in directory
        cdpoae_paths = files.find_npz_files(d, prefix='cdpoae')
        results = []
        if cdpoae_paths:
            for p in cdpoae_paths:
                logger.info('Loading DPOAE results from  %s.', p)
                cont_recording = files.load_cdpoae_recording(p)
                if cont_recording is None:
                    logger.error('Failed to load continuous DPOAE result.')
                else:
                    cdpoae_result = ContDpoaeResult(cont_recording)
                    cdpoae_result.plot(block_loop = not a)
                    results.append(cdpoae_result)
        if a:
            # start matplotlib event loop to show all figures
            plt.show()
        logger.info('All DPOAE files from  %s processed.', d)
        return

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

parser.add_argument(
    '--d',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify directory with CDPOAE recording files.'
)

parser.add_argument(
    '--a',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Show all figures simultaneously (when plotting from directory).'
)


if __name__ == "__main__":
    # Entry point for script execution
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
