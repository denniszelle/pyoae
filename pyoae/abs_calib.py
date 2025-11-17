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

from pyoae import get_logger
from pyoae import soae
from pyoae.device.device_config import DeviceConfig
from pyoae.msrmt_context import MsrmtContext
from pyoae.plot_context import SpectralPlotContext
from pyoae.soae import SoaeRecorder
from pyoae.sync import MsrmtState, SyncMsrmt

logger = get_logger()


def max_ref_pressure(ref_in: float, ref_db_spl: float = 94.0) -> float:
    """Computes the maximum peak pressure from reference calibrator.

    Args:
        ref_in: measured input level at reference frequency in dBFS_RMS
        ref_db_spl: output level of reference calibrator in dB SPL

    Returns:
        maximum peak pressure in muPa at digital full scale (1.0)

    Notes:
        Both input values are treated as RMS values.
    """
    y = 10**(ref_in/20)  # this is RMS
    x = 20 * 10**(ref_db_spl/20)  # this is muPa RMS
    return x/y


def plot_offline(
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    plot_ctx: SpectralPlotContext
) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.
        plot_ctx: Parameters and instances to control plots.

    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return
    recorded_signal, spectrum = soae.get_results(sync_msrmt, msrmt_ctx)
    _, ax_time, line_time, ax_spec, line_spec = soae.setup_plot(
        sync_msrmt.recording_data.msrmt_duration,
        sync_msrmt.recording_data.fs,
        msrmt_ctx.block_size,
        plot_ctx.live_display_duration,
        msrmt_ctx.input_trans_fun is not None
    )
    line_time.set_xdata(np.arange(len(recorded_signal))/msrmt_ctx.fs)
    line_time.set_ydata(recorded_signal)
    ax_time.set_xlim(0, sync_msrmt.recording_data.msrmt_duration)
    ax_time.set_xlabel("Recording Time (s)")

    spec_min = min(spectrum[1:])
    spec_max = max(spectrum)
    # convert dBFS to dBFS_RMS
    spec_rms = spec_max - 20 * np.log10(np.sqrt(2))
    max_input_pressure = max_ref_pressure(spec_rms)

    padding = 15  # dB of padding on top and bottom
    ax_spec.set_ylim(spec_min - padding, spec_max + padding)
    line_spec.set_ydata(spectrum)

    calib_result = (
        f'Reference input: {spec_max:.2f} dBFS '
        f'-> Input pressure at full scale: {max_input_pressure: .2f}.'
    )
    logger.info(calib_result)
    if spec_max < -20:
        logger.warning(
            'Calibration result might be invalid! '
            'Input level probably too low.'
        )
    plt.title(calib_result)
    plt.tight_layout()
    plt.show()


class AbsCalibRecorder(SoaeRecorder):
    """Class to manage an absolute calibration recording."""

    def record(self) -> None:
        """Starts the recording."""

        self.logger.info("Starting absolute calibration...")
        self.msrmt.start_msrmt(
            soae.update_msrmt,
            self.msrmt_ctx,
            self.plot_ctx
        )

        # Plot offline results after measurement
        self.logger.info(
            'Showing offline results. Please close window to continue.'
        )
        plot_offline(self.msrmt, self.msrmt_ctx, self.plot_ctx)

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
            self.msrmt_ctx
        )
        np.savez(
            save_path,
            spectrum=spectrum,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate
        )
        self.logger.info("Calibration saved to %s.", save_path)
