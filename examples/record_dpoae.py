"""Example script to record continuous DPOAEs.

This script performs a real-time acquisition of continuous distortion-product
otoacoustic emissions (DPOAEs) using two primary tones (f1 and f2).
The recorded microphone signal is analyzed in real time, and live plots of
the time signal and spectrum are displayed.

Key steps performed by this script:
- Configures and generates sinusoidal stimulus signals
- Sets up real-time visualization of signal and spectrum
- Runs a synchronized two-channel playback and recording loop
- Applies artifact rejection based on RMS thresholding
- Saves the measurement data to a `.npz` file for further analysis

This script is intended as an example and a starting point for continuous
DPOAE measurements.

Run the following command from the project root directory to start:
    python -m examples.live_cdpoae_script

Note:
    Sound device IDs or names should be known beforehand and can be obtained
      using the display_devices script.

    Users should modify parameters defined in the `device_config` module
      to match their specific hardware and experimental setup.
"""

from datetime import datetime
import os

import numpy as np
from scipy import signal

from examples import device_config
from pyoae import cdpoae
from pyoae.cdpoae import DpoaePlotInfo, DpoaeUpdateInfo
from pyoae.sync import (
    HardwareData,
    PeriodicRampSignal,
    RecordingData,
    SyncMsrmt
)


UPDATE_INTERVAL: float = 100.
"""Interval in milliseconds for updating the plots."""

LIVE_DISPLAY_DURATION: float = 100.
"""Duration that will be displayed in the live time plot in milliseconds."""

LEVEL2: float = -35.
"""Level in dB re full-scale of the second primary tone."""

LEVEL1: float = -15.
"""Level in dB re full-scale of the first primary tone."""

F2: float = 8000.
"""Frequency of the second primary tone in Hz."""

F2F1_RATIO: float = 1.2
"""Primary-tone frequency ratio of f2/f1."""

RECORDING_DURATION: float = 10.
"""Total recording duration in seconds."""

BLOCK_DURATION: float = .1
"""Duration of a single measurement block in seconds."""

ARTIFACT_REJECTION_THR = 1.8
"""Threshold for artifact rejection

Reject blocks with root-mean-squared(RMS) larger than
artifact_rejection_thr * average RMS
"""


def main() -> None:
    """Main function executing the continuous DPOAE measurement."""

    amplitude2 = 10**(LEVEL2/20)
    amplitude1 = 10**(LEVEL1/20)
    f2 = cdpoae.correct_frequency(
        F2, BLOCK_DURATION
    )
    f1 = cdpoae.correct_frequency(
        f2/F2F1_RATIO, BLOCK_DURATION
    )

    ramp_len = int(BLOCK_DURATION * device_config.FS/2)
    ramp = signal.get_window('hann', ramp_len*2)[:ramp_len].astype(np.float32)

    # Generate output signals
    samples = np.arange(
        int(device_config.FS * BLOCK_DURATION), dtype=np.float32
    )
    t = samples / device_config.FS
    play_signal1 = amplitude1 * np.sin(2*np.pi*f1*t).astype(np.float32)
    signal1 = PeriodicRampSignal(
        play_signal1,
        int(RECORDING_DURATION * device_config.FS),
        ramp
    )

    play_signal2 = amplitude2 * np.sin(2*np.pi*f2*t).astype(np.float32)
    signal2 = PeriodicRampSignal(
        play_signal2,
        int(RECORDING_DURATION * device_config.FS),
        ramp
    )

    # Create plot
    fig, ax_time, line_time, ax_spec, line_spec = cdpoae.setup_plot(
        RECORDING_DURATION,
        device_config.FS,
        int(BLOCK_DURATION * device_config.FS),
        (f1*0.6, f2*1.5),
        LIVE_DISPLAY_DURATION
    )

    # Create object with infos needed for updates
    dpoae_info = DpoaePlotInfo(
        fig,
        ax_time,
        line_time,
        ax_spec,
        line_spec,
        UPDATE_INTERVAL,
        LIVE_DISPLAY_DURATION
    )
    update_info = DpoaeUpdateInfo(
        dpoae_info,
        device_config.FS,
        int(BLOCK_DURATION * device_config.FS),
        f1,
        f2,
        ARTIFACT_REJECTION_THR
    )

    # Start measurement
    print("Starting recording...")
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
        device_config.OUTPUT_DEVICE,
    )
    msrmt = SyncMsrmt(
        rec_data,
        hw_data,
        [signal1, signal2]
    )
    msrmt.start_msrmt(cdpoae.start_plot, update_info)

    cdpoae.plot_offline(msrmt, update_info)

    # Save measurement to file.
    save_path = os.path.join(
        os.getcwd(),
        'measurements'
    )
    os.makedirs(save_path, exist_ok=True)
    cur_time = datetime.now()
    time_stamp = cur_time.strftime("%y%m%d-%H%M%S")
    file_name = 'cdpoae_msrmt_'+ time_stamp
    save_path = os.path.join(save_path, file_name)
    recorded_signal, spectrum = cdpoae.get_results(msrmt, update_info)
    np.savez(save_path,
        spectrum=spectrum,
        recorded_signal=recorded_signal,
        samplerate=device_config.FS,
        recorded_sync=msrmt.live_msrmt_data.sync_recorded
    )
    print(f"Saved measurement to {save_path}")


if __name__ == "__main__":
    # Entry point for script execution
    main()
