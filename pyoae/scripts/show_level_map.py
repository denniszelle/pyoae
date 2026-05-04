"""Script to show a pulsed DPOAE level map.

Run the following command from the project root directory to
plot all pulsed DPOAE recordings in a specified directory:

    show_level_map --d 'measurements'

Command-line arguments:
    --d: path to directory with multiple result files to be shown

"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyoae import files
from pyoae import pyoae_logger
from pyoae.dsp.pdsp import PulseDpoaeResult

try:
    matplotlib.use('qtagg')
except ImportError:
    print('No Qt Bindings found. Using default backend instead.')


logger = pyoae_logger.get_pyoae_logger('PyOAE DPOAE Results')

def show_lm(results: list[PulseDpoaeResult]) -> None:
    """Creates a level map plot from the list of DPOAE results."""

    num_measurements = len(results)
    samplerate = results[0].recording['samplerate']

    level2s = []
    level1s = []
    p_dps = []
    avg_signals = []
    avg_envelopes = []

    for result_i in results:
        level2s.append(result_i.recording['level2'])
        level1s.append(result_i.recording['level1'])
        p_dps.append(np.max(result_i.dpoae_envelope))
        avg_signals.append(result_i.dpoae_signal)
        avg_envelopes.append(result_i.dpoae_envelope)

    # Create figure and axes
    fig = plt.figure(figsize=(10,7))
    ax_lm_2d = fig.add_subplot(2, 2, 2)
    ax_lm_3d = fig.add_subplot(2,2,1, projection="3d")
    ax_signal = fig.add_subplot(2,1,2)

    # Setup 3D level map plot
    lm_3d = ax_lm_3d.scatter(
        np.asarray(level2s),
        np.asarray(level1s),
        np.asarray(p_dps), # type: ignore
        c=np.asarray(p_dps),
        cmap='turbo',
        s=80,
        picker=5
    )
    ax_lm_3d.set_xlabel('Level 2 (dB SPL)')
    ax_lm_3d.set_ylabel('Level 1 (dB SPL)')
    ax_lm_3d.set_zlabel('p_dp (muPa)')
    ax_lm_3d.set_title('Level Map')

    # Setup 2D level map plot
    lm_2d = ax_lm_2d.scatter(
        np.asarray(level2s),
        np.asarray(level1s),
        c=np.asarray(p_dps),
        cmap='turbo',
        s=80,
        picker=5
    )
    ax_lm_2d.set_xlabel('Level 2 (dB SPL)')
    ax_lm_2d.set_ylabel('Level 1 (dB SPL)')


    # Setup DPOAE time signal plot
    line_avg, = ax_signal.plot([], [])
    line_env, = ax_signal.plot([], [])
    ax_signal.set_xlabel('Time (ms)')
    ax_signal.set_ylabel('Amplitude (muPa)')
    ax_signal.set_title('Time Signal')

    # Setup measurement update
    def on_pick(event):
        """Update the time signal plot and mark the selected measurement"""
        idx = event.ind[0]
        avg_signal = avg_signals[idx]
        avg_envelope = avg_envelopes[idx]
        t_vec = np.arange(len(avg_signal)) / samplerate * 1E3

        # Update bottom plot
        line_avg.set_data(t_vec, avg_signal)
        line_env.set_data(t_vec, avg_envelope)
        ax_signal.set_xlim(t_vec.min(), t_vec.max())
        ax_signal.set_ylim(avg_envelope.min()-20, avg_envelope.max()+20)
        ax_signal.set_title(
            f'Signal for Level 2: {level2s[idx]} dB SPL, Level1: {level1s[idx]} dB SPL'
        )

        # Highlight selected point in 3D
        sizes = np.full(num_measurements, 80)
        sizes[idx] = 250

        lm_2d.set_sizes(sizes)
        lm_3d.set_sizes(sizes)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.tight_layout()
    plt.show()



def main(d: str = '') -> None:
    """Main function a level map."""

    logger.info('Display a level map from pulse DPOAE recordings.')

    _, d_name = os.path.split(d)
    logger.info('Directory name: %s', d_name)

    results = []
    if d:
        logger.info('Loading pDPOAE files from %s.', d)
        # plot results from all pDPOAE measurement files in directory
        pdpoae_paths = files.find_npz_files(d, prefix='pdpoae')

        if pdpoae_paths:
            for p in pdpoae_paths:
                # logger.info('Loading DPOAE results from  %s.', p)
                pulse_recording = files.load_pdpoae_recording(p)
                if pulse_recording is None:
                    logger.error('Failed to load pulsed DPOAE result.')
                else:
                    pdpoae_result = PulseDpoaeResult(pulse_recording)
                    pdpoae_result.set_analytic_results()
                    results.append(pdpoae_result)

        logger.info('All DPOAE files from %s loaded.', d)

    if not results:
        logger.error('Failed to load pulsed DPOAE result.')
        return

    show_lm(results)


parser = argparse.ArgumentParser(description='PyOAE Pulsed Level Map')
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
