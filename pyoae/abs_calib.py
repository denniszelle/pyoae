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
from pyoae.soae import SoaeRecorder
from pyoae.sync import MsrmtState

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


class AbsCalibRecorder(SoaeRecorder):
    """Class to manage an absolute calibration recording."""

    def record(self) -> None:
        """Starts the recording."""

        self.logger.info("Starting absolute calibration...")
        self.msrmt.run_msrmt()

        # Plot offline results after measurement
        self.logger.info(
            'Showing offline results. Please close window to continue.'
        )
        self.plot_offline()

    def plot_offline(self) -> None:
        """Shows the final results in a polished plot.

        This method overwrites that of `SoaeRecorder`.
        """
        if self.msrmt.state != MsrmtState.FINISHED:
            return
        recorded_signal, spectrum = self.get_results()

        self._plot_offline(recorded_signal, spectrum)
        # convert dBFS to dBFS_RMS
        spec_max = max(spectrum)
        spec_rms = spec_max - 20 * np.log10(np.sqrt(2))
        max_input_pressure = max_ref_pressure(spec_rms)

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
        recorded_signal, spectrum = self.get_results()
        np.savez(
            save_path,
            spectrum=spectrum,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate
        )
        self.logger.info("Calibration saved to %s.", save_path)
