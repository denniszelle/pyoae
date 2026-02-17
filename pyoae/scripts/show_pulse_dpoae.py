"""Script to show pulsed DPOAE data.

This script shows pulsed DPOAE response(s) in the time domain
from recording(s) using short-pulsed primary tones.


Run the following command from the project root directory to
plot results from a single measurement:

    show_pulse_dpoae --file 'pdpoae_msrmt_251021-143047_2000_45.npz'

Run the following command from the project root directory to
plot all pulsed DPOAE recordings in a specified directory:

    show_pulse_dpoae --d 'measurements'

Command-line arguments:
    --file: path to result file to be shown,
    --d: path to directory with multiple result files to be shown
    --a: show all figures simultaneously at the end of the processing

"""

import argparse

from matplotlib import pyplot as plt

from pyoae import files
from pyoae import pyoae_logger
from pyoae.dsp.processing import PulseDpoaeResult


logger = pyoae_logger.get_pyoae_logger('PyOAE Pulse-DPOAE Results')

def main(file: str = '', d: str = '', a: bool = False) -> None:
    """Main function visualizing a pulsed DPOAE response."""

    logger.info('Display of results from pulsed DPOAE recording.')

    pdpoae_result = None

    if d:
        logger.info('Loading DPOAE files from %s.', d)
        # plot results from all pDPOAE measurement files in directory
        pdpoae_paths = files.find_npz_files(d, prefix='pdpoae')
        # keep result objects in list to show all figures at the end
        results = []
        if pdpoae_paths:
            for p in pdpoae_paths:
                logger.info('Loading DPOAE results from  %s.', p)
                pulsed_recording = files.load_pdpoae_recording(p)
                if pulsed_recording is None:
                    logger.error('Failed to load pulsed DPOAE result.')
                else:
                    pdpoae_result = PulseDpoaeResult(pulsed_recording)
                    pdpoae_result.plot(block_loop = not a)
                    results.append(pdpoae_result)
        if a:
            # start matplotlib event loop to show all figures
            plt.show()
        logger.info('All DPOAE files from  %s processed.', d)
        return

    if file:
        logger.info('Loading DPOAE results from  %s.', file)
        pulsed_recording = files.load_pdpoae_recording(file)
        if pulsed_recording is not None:
            pdpoae_result = PulseDpoaeResult(pulsed_recording)

    if pdpoae_result is None:
        logger.error('Failed to load pulse DPOAE result.')
        return

    pdpoae_result.plot()


parser = argparse.ArgumentParser(description='PyOAE Pulse-DPOAE Result')
parser.add_argument(
    '--file',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify pDPOAE recording file.'
)

parser.add_argument(
    '--d',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify directory with pDPOAE recording files.'
)

parser.add_argument(
    '--a',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Show all figures simultaneously (when plotting from directory).'
)


def run_cli() -> None:
    """Run main with console arguments"""
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)


if __name__ == "__main__":
    # Entry point for console module execution
    run_cli()
