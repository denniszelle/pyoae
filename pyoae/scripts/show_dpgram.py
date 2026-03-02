"""Script to show a DP gram

Run the following command from the project root directory to
plot all continuous DPOAE recordings in a specified directory:

    show_dpgram --d 'measurements'

Command-line arguments:
    --d: path to directory with multiple result files to be shown

"""

import argparse
import os

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from pyoae import files
from pyoae import pyoae_logger
from pyoae.dsp.cdsp import ContDpoaeResult

try:
    matplotlib.use('qtagg')
except ImportError:
    print('No Qt Bindings found. Using default backend instead.')


logger = pyoae_logger.get_pyoae_logger('PyOAE DPOAE Results')

NUM_NOISE_BINS_PER_SIDE = 5


def show_dp_gram(results: list[ContDpoaeResult], label: str = '') -> None:
    """Creates a DP gram from the list of DPOAE results."""
    primary2_frequencies = []
    dp_levels = []
    noise_levels = []
    primary2_levels = []
    primary1_levels = []

    for result in results:
        samplerate = result.recording['samplerate']
        num_samples = result.recording['num_block_samples']
        df = samplerate/num_samples

        f1 = result.recording['f1']
        f2 = result.recording['f2']
        fdp = 2*f1 - f2
        idx_dp = int(fdp/df)
        level_dp = result.dpoae_spectrum[idx_dp]

        noise_idx = np.concatenate(
            [
                np.arange(idx_dp-NUM_NOISE_BINS_PER_SIDE,idx_dp),
                np.arange(idx_dp+1, idx_dp+NUM_NOISE_BINS_PER_SIDE+1)
            ],
        )

        level_noise = np.mean(result.dpoae_spectrum[noise_idx])

        primary1_levels.append(result.recording['level1'])
        primary2_levels.append(result.recording['level2'])
        dp_levels.append(level_dp)
        noise_levels.append(level_noise)
        primary2_frequencies.append(f2)

    # convert to NumPy array
    dp_levels = np.array(dp_levels)
    noise_levels = np.array(noise_levels)
    primary1_levels = np.array(primary1_levels)
    primary2_levels = np.array(primary2_levels)
    primary2_frequencies = np.array(primary2_frequencies)

    ul2 = np.unique(primary2_levels)
    for l2 in ul2:
        l2_idx = primary2_levels == l2
        plt.plot(
            primary2_frequencies[l2_idx]*1E-3,
            dp_levels[l2_idx],
            marker='.',
            color=[0.8, 0 ,0],
            markersize=3,
            linewidth=0.5
        )
        plt.plot(
            primary2_frequencies[l2_idx]*1E-3,
            noise_levels[l2_idx],
            '-',
            marker='^',
            markersize=2,
            color='k',
            linewidth=0.5
        )
        plt.xlabel('f2 (kHz)')
        plt.ylabel('L_DP (dB SPL)')
        plt.yticks(np.arange(-50, 70, 10), minor=True)
        plt.ylim(-50, 60)
        plt.xticks(
            [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10]
        )
        plt.xlim(0.2, 10.5)
        plt.grid(which='both')
        plt.xscale='log'
        plt.title(f'{label} Level2: {l2:.1f} dB SPL', fontsize=10)
        plt.tight_layout()
        plt.show()


def main(d: str = '') -> None:
    """Main function visualizing a cDP gram."""

    logger.info('Display DP gram from continuous DPOAE recordings.')

    _, d_name = os.path.split(d)
    logger.info('Directory name: %s', d_name)

    results = []
    if d:
        logger.info('Loading DPOAE files from %s.', d)
        # plot results from all cDPOAE measurement files in directory
        cdpoae_paths = files.find_npz_files(d, prefix='cdpoae')

        if cdpoae_paths:
            for p in cdpoae_paths:
                # logger.info('Loading DPOAE results from  %s.', p)
                cont_recording = files.load_cdpoae_recording(p)
                if cont_recording is None:
                    logger.error('Failed to load continuous DPOAE result.')
                else:
                    cdpoae_result = ContDpoaeResult(cont_recording)
                    results.append(cdpoae_result)

        logger.info('All DPOAE files from %s loaded.', d)

    if not results:
        logger.error('Failed to load continuous DPOAE result.')
        return

    show_dp_gram(results, label=d_name)


parser = argparse.ArgumentParser(description='PyOAE Cont. DP Gram')
parser.add_argument(
    '--d',
    default=argparse.SUPPRESS,
    type=str,
    help='Specify directory with CDPOAE recording files.'
)


def run_cli() -> None:
    """Run main with console arguments"""
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)


if __name__ == "__main__":
    # Entry point for console module execution
    run_cli()
