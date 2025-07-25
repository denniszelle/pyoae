"""Example script to record continuous DPOAEs.

This script performs a real-time acquisition of continuous distortion-product
otoacoustic emissions (DPOAEs) using two primary tones (f1 and f2).
The recorded microphone signal is analyzed in real time, and live plots of
the time signal and spectrum are displayed.

Key steps performed by this script:
- Configures and generates sinusoidal stimulus signals
- Sets up real-time visualization of signal and spectrum
- Runs a synchronized two-channel playback and recording loop
- Applies artifact rejection based on an RMS threshold
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

from examples import device_config
from pyoae import cdpoae
from pyoae.cdpoae import DpoaePlotInfo, DpoaeUpdateInfo
from pyoae.sync import (
    HardwareData,
    PeriodicSignal,
    PeriodicRampSignal,
    RecordingData,
    SyncMsrmt
)

UPDATE_INTERVAL: float = 100.
"""Interval in milliseconds for updating the plots."""

LIVE_DISPLAY_DURATION: float = 100.
"""Duration that will be displayed in the live time plot in milliseconds."""

LEVEL2: float = -35.
"""Level in dB re full scale (dBFS) of the second primary tone."""

LEVEL1: float = -25.
"""Level in dB re full scale (dBFS) of the first primary tone."""

F2: float = 3000.
"""Frequency of the second primary tone in Hz."""

F2F1_RATIO: float = 1.2
"""Primary-tone frequency ratio.

r = f2/f1

Used to calculate the frequency of the
first primary tone.
"""

NUM_AVERAGING_BLOCKS: int = 100
"""Number of blocks used for averaging.

Total recording time is
  BLOCK_DURATION * NUM_AVERAGING_BLOCKS
"""

BLOCK_DURATION: float = .1
"""Duration of a single measurement block in seconds.

Total recording time is
  BLOCK_DURATION * NUM_AVERAGING_BLOCKS
"""

ARTIFACT_REJECTION_THR = 1.8
"""Threshold for simple artifact rejection.

Reject blocks with a root-mean-square (RMS) value
exceeding ARTIFACT_REJECTION_THR * median_rms.
"""

USE_RAMP = True
"""Flag enabling/disabling fade-in/out of stimuli using a ramp.

If set to True, the script applies a ramp to fade-in the stimuli
at the start of the first data block and fade-out at the end of
the last block, respectively.
"""

RAMP_DURATION = 5
"""Duration of fade-in/out ramp in ms."""


def main() -> None:
    """Main function executing the continuous DPOAE measurement."""

    # calculate number of samples and ensure an integer number
    num_block_samples = int(device_config.FS * BLOCK_DURATION)
    # ensure block duration matches number of block samples
    block_duration = num_block_samples / device_config.FS
    if block_duration != BLOCK_DURATION:
        print(
            f'Block duration adjusted to {block_duration*1E3:.2f} ms'
        )

    amplitude2 = 10**(LEVEL2/20)
    amplitude1 = 10**(LEVEL1/20)
    f2 = cdpoae.correct_frequency(F2, num_block_samples/block_duration)
    f1 = cdpoae.correct_frequency(f2/F2F1_RATIO, block_duration)
    print(
        'Setting primary-tone frequencies: '
        f'f1: {f1:.2f} Hz, f2: {f2:.2f} Hz, '
        f'(f2/f1 = {f2/f1: .3f})'
    )
    print(
        'Setting output levels: '
        f'L1: {LEVEL1:.1f} dBFS ({amplitude1:.5f} re FS).'
        f'L2: {LEVEL2:.1f} dBFS ({amplitude2:.5f} re FS).'
    )

    # Generate output signals
    samples = np.arange(num_block_samples, dtype=np.float32)
    t = samples / device_config.FS
    play_signal1 = amplitude1 * np.sin(2*np.pi*f1*t).astype(np.float32)
    play_signal2 = amplitude2 * np.sin(2*np.pi*f2*t).astype(np.float32)

    num_total_recording_samples = num_block_samples * NUM_AVERAGING_BLOCKS

    if USE_RAMP:
        ramp_len = int(RAMP_DURATION*1E-3 * device_config.FS)
        ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
        ramp = ramp.astype(np.float32)
        signal1 = PeriodicRampSignal(
            play_signal1,
            num_total_recording_samples,
            ramp
        )
        signal2 = PeriodicRampSignal(
            play_signal2,
            num_total_recording_samples,
            ramp
        )
    else:
        signal1 = PeriodicSignal(play_signal1, num_total_recording_samples)
        signal2 = PeriodicSignal(play_signal2, num_total_recording_samples)

    # Create plot
    total_recording_duration = num_total_recording_samples / device_config.FS
    fig, ax_time, line_time, ax_spec, line_spec = cdpoae.setup_plot(
        total_recording_duration,
        device_config.FS,
        num_block_samples,
        (f1*0.6, f2*1.5),
        LIVE_DISPLAY_DURATION
    )

    # Create object with info needed for updates
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
        num_block_samples,
        f1,
        f2,
        ARTIFACT_REJECTION_THR
    )

    rec_data = RecordingData(
        float(device_config.FS),
        float(total_recording_duration),
        num_total_recording_samples,
        num_block_samples,
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

    print("Starting recording...")
    # `start_msrmt` starts the application loop
    msrmt.start_msrmt(cdpoae.start_plot, update_info)
    # Plot all data and final result after user has
    # closed the live-measurement window.
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
