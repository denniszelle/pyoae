"""Classes and functions to perform absolute calibration.

This module contains classes and utility functions used to
perform an absolute calibration of the input channel. The
measurement is based on an SOAE recording.

Key functionalities include:
- Setup of live plots for the measurement
- Robust, asynchronous spectrum estimation

This module is not intended to be run directly.
"""

from datetime import datetime
import os

from matplotlib import pyplot as plt
import numpy as np

from pyoae import soae
from pyoae.device.device_config import DeviceConfig
from pyoae.soae import SoaeRecorder, SoaeUpdateInfo
from pyoae.sync import MsrmtState, SyncMsrmt


def max_output_pressure(ref_in: float, ref_db_spl: float) -> float:
    """Computes the maximum output peak pressure from reference calibrator.

    Args:
        ref_in: measured input level at reference frequency in dBFS
        ref_db_spl: output level of reference calibrator in dB SPL

    Returns:
        maximum peak pressure in muPa at digital full scale (1.0)

    Notes:
        Both input values are treated as RMS values.
    """
    y = 10**(ref_in/20)  # this is RMS
    x = 20 * 10**(ref_db_spl/20)  # this is muPa RMS
    return x/y


def plot_offline(sync_msrmt: SyncMsrmt, info: SoaeUpdateInfo) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos and plot objects

    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return
    recorded_signal, spectrum = soae.get_results(sync_msrmt, info)
    _, ax_time, line_time, ax_spec, line_spec = soae.setup_plot(
        sync_msrmt.recording_data.msrmt_duration,
        sync_msrmt.recording_data.fs,
        info.block_size,
        info.plot_info.live_display_duration,
        info.correction_tf is not None
    )
    line_time.set_xdata(np.arange(len(recorded_signal))/info.fs)
    line_time.set_ydata(recorded_signal)
    ax_time.set_xlim(0, sync_msrmt.recording_data.msrmt_duration)
    ax_time.set_xlabel("Recording Time (s)")

    spec_min = min(spectrum[1:])
    spec_max = max(spectrum)
    padding = 15  # dB of padding on top and bottom
    ax_spec.set_ylim(spec_min - padding, spec_max + padding)
    line_spec.set_ydata(spectrum)
    plt.title(f'Maximum: {spec_max:.5}')
    plt.tight_layout()
    plt.show()


class AbsCalibRecorder(SoaeRecorder):
    """Class to manage an absolute calibration recording."""

    def record(self) -> None:
        """Starts the recording."""

        self.logger.info("Starting absolute calibration...")
        self.msrmt.start_msrmt(soae.start_plot, self.update_info)

        # Plot offline results after measurement
        plot_offline(self.msrmt, self.update_info)

    def save_recording(self) -> None:
        """Stores the measurement data in binary file."""
        # Save measurement to file.
        save_path = os.path.join(
            os.getcwd(),
            'measurements'
        )
        os.makedirs(save_path, exist_ok=True)
        cur_time = datetime.now()
        time_stamp = cur_time.strftime("%y%m%d-%H%M%S")
        file_name = 'abs_calib_msrmt_'+ time_stamp
        save_path = os.path.join(save_path, file_name)
        recorded_signal, spectrum = soae.get_results(
            self.msrmt,
            self.update_info
        )
        np.savez(
            save_path,
            spectrum=spectrum,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate
        )
        self.logger.info("Calibration saved to %s.", save_path)
