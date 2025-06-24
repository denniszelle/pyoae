"""Functions to record continuous DPOAEs

This module contains utility functions used to acquire, process
and analyze continuous distortion-product otoacoustic emissions (cDPOAE).
These functions are typically used within example scripts, but can
also be imported and reused in other analysis pipelines.

Key functionalities include:
- Setup of live plots for the measurement
- Robust, synchronous spectrum estimation

Typical usage:

```
    import pyoae.cdpoae
```

This module is not intended to be run directly.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pyoae.sync import SyncMsrmt, MsrmtState
from pyoae.calib import MicroTransferFunction


@dataclass
class DpoaePlotInfo:
    """Container with data for the measurement plot."""

    fig: Figure
    """Figure corresponding to the plot window."""

    ax_time: Axes
    """Axis object of the time plot."""

    line_time: Line2D
    """Line object for the time plot."""

    ax_spec: Axes
    """Axis object of the spectral plot."""

    line_spec: Line2D
    """Line object for the spectral plot."""

    update_interval: float
    """Interval to apply processing and plot update during measurement."""

    live_display_duration: float
    """Duration to display time domain plot in ms."""


@dataclass
class DpoaeUpdateInfo:
    """Container with data for continuous DPOAE measurement updates."""

    plot_info: DpoaePlotInfo
    """Storage for the measurement plot."""

    fs: float
    """Sampling frequency in Hz."""

    block_size: int
    """Number of samples in each block."""

    f1: float
    """Lower stimulus frequency in Hz."""

    f2: float
    """Upper stimulus frequency in Hz."""

    artifact_rejection_thr: float
    """Threshold for artifact rejection"""

    correction_tf: MicroTransferFunction | None = None
    """Microphone transfer function"""


def correct_frequency(frequency: float, segment_duration: float) -> float:
    """Corrects frequency to obtain an integer number of periods in segment."""
    periods = segment_duration*frequency
    periods = round(periods)
    return periods/segment_duration


def setup_plot(
    recording_duration: float,
    fs: float,
    block_size: int,
    frequency_range: tuple[float, float],
    live_display_duration: float,
    is_calib_available:bool=False
) -> tuple[Figure, Axes, Line2D, Axes, Line2D]:
    """Sets up the plots.

    Args:
        recording_duration: Total duration of the recording in seconds
        fs: Sampling frequency in Hz
        block_size: Size of each measurement block that is repeated
          periodically
        live_display_duration: Time domain display duration in milliseconds
        is_calib_available: Boolean whether a calibration is available to
          display sound pressure or only show raw measurement data

    Returns:
        tuple[fig, line_time, line_spec]

        - **fig**: Object containing the plots
        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal

    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes: list[Axes]
    (ax_time, ax_spec) = axes

    # Setup time plot
    x_wave = np.arange(round(recording_duration*fs), dtype=np.float32) / fs * 1E3
    y_wave = np.zeros_like(x_wave)
    line_time, = ax_time.plot(
        x_wave,
        y_wave
    )
    ax_time.set_ylim(-1, 1)
    ax_time.set_xlim(-live_display_duration, 0)
    ax_time.set_title("Recorded Waveform")
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel("Amplitude (full-scale)")

    # Setup frequency plot
    fft_freqs = np.fft.rfftfreq(block_size, 1 / fs)
    fft_vals = np.zeros(len(fft_freqs))
    line_spec, = ax_spec.plot(fft_freqs, fft_vals)
    print(f'Plotting spectral x-data with length: {len(fft_freqs)}')
    ax_spec.set_xlim(frequency_range[0], frequency_range[1])
    ax_spec.set_ylim(-50, 100)
    ax_spec.set_title("Spectrum")
    ax_spec.set_xlabel("Frequency (Hz)")
    if is_calib_available:
        ax_spec.set_ylabel('Level (dB SPL)')
    else:
        ax_spec.set_ylabel("Level (dB re full-scale)")
    return fig, ax_time, line_time, ax_spec, line_spec


def process_spectrum(
    recorded_signal: np.ndarray,
    block_size: int,
    correction_tf: MicroTransferFunction | None,
    artifact_rejection_thr:float
) -> np.ndarray:
    """Processes recorded signal to obtain spectrum.

    The signal is averaged in the time domain rejecting blocks above some
    value relative to the average RMS. Then, the spectrum is obtained from
    the averaged signal.

    Args:
        recorded_signal: float array of measurement data
        block_size: Size of each recording block in samples
        correction_tf: Transfer function of the microphone
        artifact_rejection_thr: Threshold representing a factor where blocks
          larger than factor*mean_rms are rejected

    Returns:
        Array of floats containing the averaged spectrum.

    """

    spectrum = None
    # Obtain an integer number of recorded blocks
    total_blocks = int(np.floor(len(recorded_signal)/block_size))
    block_data = recorded_signal[:total_blocks*block_size]
    # Only apply processing for at least 3 blocks
    if total_blocks > 2:
        # Reshape data into block structure removing the first and final blocks
        blocks = block_data.reshape(
            -1,
            block_size
        )[1:-1]

        # Reject RMS values larger than artifact_rejection_thr*average_rms
        rms_vals = np.sqrt(np.mean(np.square(blocks), axis=1))
        avg_rms = np.median(rms_vals)
        accepted_idc = np.where(rms_vals<artifact_rejection_thr*avg_rms)[0]

        #Average blocks
        if len(accepted_idc):
            avg = blocks[accepted_idc].mean(axis=0)
        else:
            avg = np.zeros(block_size)

        # Apply FFT, correct spectrum by microphone calibration, convert to
        # dB or dB SPL scale
        if np.sqrt(np.mean(np.square(avg))) > 0:
            spectrum = np.abs(np.fft.rfft(avg))/len(avg)*2
            if correction_tf is None:
                spectrum = 20*np.log10(spectrum)
            else:
                spectrum /= correction_tf.amplitudes
                spectrum = 20*np.log10(spectrum/20)

    if spectrum is None:
        spectrum = np.abs(np.fft.rfft(np.zeros(block_size, np.float32)))
    return spectrum


def get_results(
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo
) -> tuple[np.ndarray, np.ndarray]:
    """Processes data and returns plot results.

    If the measurement is currently running, the recorded signal is obtained
    and a synchronously averaged spectrum is estimated.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos about the measurement.

    Returns:
        tuple[recorded_signal, spectrum]

        - **recorded_signal**: Float array with the recorded signal
        - **spectrum**: Float array with the spectral estimate

    """

    # Only update while or after main measurement
    if sync_msrmt.state in [
        MsrmtState.MEASURING,
        MsrmtState.FINISHING,
        MsrmtState.FINISHED
    ]:

        recorded_signal = sync_msrmt.get_recorded_signal()

        spectrum = process_spectrum(
            recorded_signal,
            info.block_size,
            info.correction_tf,
            info.artifact_rejection_thr
        )

        return recorded_signal, spectrum

    return np.zeros(0,np.float32), np.zeros(0,np.float32)


def update_plot_data(
    recorded_signal: np.ndarray,
    spectrum: np.ndarray,
    info: DpoaeUpdateInfo
) -> tuple[Line2D, Line2D]:
    """Updates the plot data.

    Create and set the plot objects for the data.

    Args:
        recorded_signal: Float array containing the raw signal.
        spectrum: Float array containing the spectral estimate.
        info: Info object containing meta infos and plot objects

    Returns:
        tuple[line_time, line_spec]

        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal

    """

    if len(spectrum) == 0:
        return info.plot_info.line_time, info.plot_info.line_spec

    n_samples_displayed = int(info.plot_info.live_display_duration*1E-3*info.fs)

    if len(recorded_signal) < n_samples_displayed:
        return info.plot_info.line_time, info.plot_info.line_spec

    x_data = -np.flipud(np.arange(n_samples_displayed)/info.fs*1E3)

    info.plot_info.line_time.set_data(
        x_data,
        recorded_signal[-n_samples_displayed:]
    )

    info.plot_info.line_spec.set_ydata(spectrum)

    # Update y-axis limits.
    spec_min = min(spectrum[1:])
    spec_max = max(spectrum)
    padding = 15  # dB of padding on top and bottom

    if (spec_min < info.plot_info.ax_spec.get_ylim()[0]
        or spec_min > info.plot_info.ax_spec.get_ylim()[0] + 2*padding
        or spec_max > info.plot_info.ax_spec.get_ylim()[1]
        or spec_max < info.plot_info.ax_spec.get_ylim()[1] - 2*padding):

        info.plot_info.ax_spec.set_ylim(spec_min - padding, spec_max + padding)



    return info.plot_info.line_time, info.plot_info.line_spec


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo
) -> tuple[Line2D, Line2D]:
    """Processes results from data and update the plots.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos and plot objects

    Returns:
        tuple[line_time, line_spec]

        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal

    """


    del frame

    if sync_msrmt.state == MsrmtState.FINISHED:
        # plt.close(info.fig)
        return info.plot_info.line_time, info.plot_info.line_spec

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.state = MsrmtState.FINISHED

    recorded_signal, spectrum = get_results(sync_msrmt, info)

    return update_plot_data(recorded_signal, spectrum, info)


def start_plot(sync_msrmt: SyncMsrmt, info: DpoaeUpdateInfo) -> None:
    """Executes the measurement plot that is regularly updated."""
    _ = FuncAnimation(
        info.plot_info.fig,
        update_msrmt,
        fargs=(sync_msrmt, info,),
        interval=info.plot_info.update_interval,
        blit=False,
        cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()
    if sync_msrmt.state not in [MsrmtState.FINISHING, MsrmtState.FINISHED]:
        sync_msrmt.state = MsrmtState.CANCELED

def plot_offline(sync_msrmt: SyncMsrmt, info: DpoaeUpdateInfo) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.

    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return
    recorded_signal, spectrum = get_results(sync_msrmt, info)
    _, ax_time, line_time, ax_spec, line_spec = setup_plot(
        sync_msrmt.recording_data.msrmt_duration,
        sync_msrmt.recording_data.fs,
        info.block_size,
        (info.f1*0.6, info.f2*1.5),
        info.plot_info.live_display_duration
    )
    line_time.set_xdata(np.arange(len(recorded_signal))/info.fs)
    line_time.set_ydata(recorded_signal)
    ax_time.set_xlim(0, sync_msrmt.recording_data.msrmt_duration)

    spec_min = min(spectrum[1:])
    spec_max = max(spectrum)
    padding = 15  # dB of padding on top and bottom
    ax_spec.set_ylim(spec_min - padding, spec_max + padding)
    line_spec.set_ydata(spectrum)
    plt.tight_layout()
    plt.show()
