"""Example script to record SOAEs (Spontaneous Otoacoustic Emissions).

This script demonstrates how to acquire and visualize SOAE data in real-time.
It uses a synchronized playback and recording setup. A live time-domain
and frequency-domain plot is shown during the measurement.

Key steps performed by this script:
- Configures signal generators and buffer parameters
- Sets up real-time plots for signal and spectrum display
- Runs a synchronized measurement using input and output sound devices
- Saves the recorded signal and sampling rate to a .npz file

This script is intended to be used as an example and entry point for
SOAE measurements.

Run the following command from the project root directory to start:
    python -m examples.live_soae_script

Note:
    Sound device IDs or names should be known beforehand and can be obtained
      using the display_devices script.

    Users should modify parameters defined in the `device_config` module
      to match their specific hardware and experimental setup.
"""

from datetime import datetime
import os

import numpy as np

from examples import device_config
from pyoae import soae
from pyoae.soae import SoaePlotInfo, SoaeUpdateInfo
from pyoae.sync import (
    HardwareData,
    PeriodicSignal,
    RecordingData,
    SyncMsrmt
)


UPDATE_INTERVAL: float = 100.
"""Interval in milliseconds for updating the plots."""

LIVE_DISPLAY_DURATION = 100.
"""Duration that will be displayed in the live-time plot in milliseconds."""

RECORDING_DURATION: float = 15.
"""Total recording duration in seconds"""

BLOCK_DURATION: float = 1.
"""Duration of a single measurement block in seconds.

This also determines the frequency resolution in the measurement."""

ARTIFACT_REJECTION_THR = 1.8
"""Threshold for artifact rejection.

Reject blocks with root-mean-squared(RMS) larger than
artifact_rejection_thr * average RMS
"""


def main() -> None:
    """Main function executing the SOAE measurement."""

    # Generate output signals
    signal1 = PeriodicSignal(
        np.zeros(device_config.DEVICE_BUFFER_SIZE, dtype=np.float32)
    )
    signal2 = PeriodicSignal(
        np.zeros(device_config.DEVICE_BUFFER_SIZE, dtype=np.float32)
    )

    # Setup live plot
    fig, ax_time, line_time, ax_spec, line_spec = soae.setup_plot(
        RECORDING_DURATION,
        device_config.FS,
        int(BLOCK_DURATION * device_config.FS),
        LIVE_DISPLAY_DURATION
    )

    # Define info object used during measurement
    soae_info = SoaePlotInfo(
        fig,
        ax_time,
        line_time,
        ax_spec,
        line_spec,
        UPDATE_INTERVAL,
        LIVE_DISPLAY_DURATION
    )
    update_info = SoaeUpdateInfo(
        soae_info,
        device_config.FS,
        int(BLOCK_DURATION * device_config.FS),
        ARTIFACT_REJECTION_THR
    )

    # Prepare measurement
    rec_data = RecordingData(
        float(device_config.FS),
        float(RECORDING_DURATION),
        int(RECORDING_DURATION * device_config.FS),
        device_config.DEVICE_BUFFER_SIZE
    )
    hw_data = HardwareData(
        2,
        2,
        device_config.INPUT_DEVICE,
        device_config.OUTPUT_DEVICE
    )

    # Start measurement
    print("Starting playback + recording...")
    msrmt = SyncMsrmt(
        rec_data,
        hw_data,
        [signal1, signal2]
    )
    msrmt.start_msrmt(soae.start_plot, update_info)

    # Plot offline results after measurement
    soae.plot_offline(msrmt, update_info)

    # Save measurement to file.
    save_path = os.path.join(
        os.getcwd(),
        'measurements'
    )
    os.makedirs(save_path, exist_ok=True)
    cur_time = datetime.now()
    time_stamp = cur_time.strftime("%y%m%d-%H%M%S")
    file_name = 'soae_msrmt_'+ time_stamp
    save_path = os.path.join(save_path, file_name)
    recorded_signal, spectrum = soae.get_results(msrmt, update_info)
    np.savez(
        save_path,
        spectrum=spectrum,
        recorded_signal=recorded_signal,
        samplerate=device_config.FS
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    # Entry point for script execution
    main()
