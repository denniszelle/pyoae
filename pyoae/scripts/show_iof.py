"""Script to show a DPOAE input/output function.

Run the following command from the project root directory to
plot all continuous DPOAE recordings in a specified directory:

    show_iof --d 'measurements'

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
from pyoae.dsp.continuous_dpoae import ContDpoaeResult

try:
    matplotlib.use('qtagg')
except ImportError:
    print('No Qt Bindings found. Using default backend instead.')


logger = pyoae_logger.get_pyoae_logger('PyOAE DPOAE Results')

NUM_NOISE_BINS_PER_SIDE = 5


def show_iof(results: list[ContDpoaeResult], label: str = '') -> None:
    """Creates an Input/Output function from the list of DPOAE results."""
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
    dp_amplitudes = 20 * 10 ** (dp_levels/20)
    noise_levels = np.array(noise_levels)
    noise_amplitudes = 20 * 10 ** (noise_levels/20)
    primary1_levels = np.array(primary1_levels)
    primary2_levels = np.array(primary2_levels)
    primary2_frequencies = np.array(primary2_frequencies)

    uf2 = np.unique(primary2_frequencies)
    for f2 in uf2:
        f2_idx = primary2_frequencies == f2
        plt.plot(
            primary2_levels[f2_idx],
            dp_amplitudes[f2_idx],
            marker='.',
            color=[0.8, 0 ,0],
            markersize=3,
            linewidth=0.5
        )
        plt.plot(
            primary2_levels[f2_idx],
            noise_amplitudes[f2_idx],
            '-',
            marker='^',
            markersize=2,
            color='k',
            linewidth=0.5
        )

        plt.xlabel('L2 (dB SPL)')
        plt.ylabel('p_DP (muPa_RMS)')
        plt.yticks(np.arange(0, 200, 50), minor=True)
        plt.ylim(0, 100)
        plt.xlim(0, 90)
        plt.grid(which='both')
        plt.title(f'{label} f2: {f2:.1f} kHz', fontsize=10)
        plt.tight_layout()
        plt.show()


def main(d: str = '') -> None:
    """Main function visualizing the IOF."""

    logger.info('Display input/output functions from continuous DPOAE recordings.')

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

    show_iof(results, label=d_name)


parser = argparse.ArgumentParser(description='PyOAE Cont. Input/Output Function')
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
