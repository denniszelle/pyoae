"""Classes and functions to record SOAE.

This module contains classes and utility functions used to
acquire, process, and analyze Spontaneous Otoacoustic Emissions
(SOAE). These functions are typically used within example scripts,
but can also be imported and reused in other analysis pipelines.

Key functionalities include:
- Setup of live plots for the measurement
- Robust, asynchronous spectrum estimation

Typical usage:

```
    from pyoae.soae import SoaeRecorder
    msrmt_params = files.load_soae_protocol(protocol)
    soae_recorder = SoaeRecorder(msrmt_params)
    soae_recorder.record()
```

This module is not intended to be run directly.
"""

from dataclasses import dataclass
from datetime import datetime
from logging import Logger
import os
from typing import Literal

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import scipy.signal

from pyoae import get_logger
from pyoae import helpers
from pyoae.calib import MicroTransferFunction
from pyoae.device.device_config import DeviceConfig
from pyoae.protocols import SoaeMsrmtParams
from pyoae.signals import PeriodicSignal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


logger = get_logger()


@dataclass
class SoaePlotInfo:
    """Container with data for the measurement plot."""

    fig: Figure
    """Figure corresponding to the plot window."""

    ax_time: Axes
    """Axis object for the time plot."""

    line_time: Line2D
    """Line object for the time plot."""

    ax_spec: Axes
    """Axis object for the spectral plot."""

    line_spec: Line2D
    """Line object for the spectral plot."""

    update_interval: float
    """Interval to apply processing and plot update during measurement in ms."""

    live_display_duration: float
    """Duration to display time domain plot in ms."""


@dataclass
class SoaeUpdateInfo:
    """Object to store data used in measurement updates."""

    plot_info: SoaePlotInfo
    """Store data for the measurement plot."""

    fs: float
    """Sampling frequency in Hz."""

    block_size: int
    """Number of samples in each block."""

    artifact_rejection_thr: float
    """Proportion of values to trim to reduce artifact influence."""

    correction_tf: MicroTransferFunction | None = None
    """Transfer function of the microphone."""


def setup_plot(
    recording_duration: float,
    fs: float,
    window_size: int,
    live_display_duration: float,
    is_calib_available:bool=False
) -> tuple[Figure, Axes, Line2D, Axes, Line2D]:
    """Sets up the plots.

    Args:
        recording_duration: Total duration of the recording in seconds
        fs: Sampling frequency in Hz
        window_size: Number of samples of the welch spectral estimation window
        live_display_duration: Time domain display duration in milliseconds
        is_calib_available: Boolean whether a calibration is available to
          display sound pressure or only show raw measurement data

    Returns:
        fig: Object containing the plots
        line_time: Line object with the time-domain data of the signal
        line_spec: Line object with the spectral data of the signal

    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes: list[Axes]
    (ax_time, ax_spec) = axes

    # Set up time plot
    x_wave = np.arange(round(recording_duration*fs), dtype=np.float32) / fs
    y_wave = np.zeros_like(x_wave)
    line_time, = ax_time.plot(
        x_wave,
        y_wave
    )
    ax_time.set_ylim(-1, 1)
    ax_time.set_xlim(-live_display_duration, 0)
    ax_time.set_title("Recorded Waveform")
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel("Amplitude (re full scale)")

    # Set up frequency plot
    fft_frequencies = np.fft.rfftfreq(window_size, 1 / fs)
    fft_values = np.zeros(len(fft_frequencies))
    line_spec, = ax_spec.plot(fft_frequencies, fft_values)
    ax_spec.set_xlim(500, 20*1E3)
    ax_spec.set_xscale('log')
    ax_spec.set_title("Spectrum")
    ax_spec.set_xlabel("Frequency (Hz)")
    if is_calib_available:
        ax_spec.set_ylabel('Level (dB SPL)')
        ax_spec.set_ylim(-50, 50)
    else:
        ax_spec.set_ylim(-150, 0)
        ax_spec.set_ylabel("Level (dBFS)")

    return fig, ax_time, line_time, ax_spec, line_spec


def welch_artifact_rejection(
    x: npt.NDArray[np.float32],
    fs: float,
    window: str = 'hann',
    window_samples: int = 256,
    overlap_samples: int | None = None,
    detrend: Literal['linear', 'constant'] = 'constant',
    artifact_rejection_thr: float = 1.8
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Estimates a welch spectrum with basic measurement artifact rejection.

    Args:
        x: Float array as time-domain signal
        fs: Sampling frequency of the measurement
        window: Type of the window (see scipy.signal.get_window)
        window_samples: Samples for each window
        overlap_samples: Window overlap samples, window_samples/2 by default
        detrend: Remove linear or constant trend beforehand
        artifact_rejection_thr: Threshold representing a factor where blocks
          larger than factor*mean_rms are rejected

    Returns:
        tuple[frequencies, spec_avg]

        - **frequencies**: Float array containing the corresponding frequencies
            for the magnitudes.
        - **spec_avg**: Float array with averaged spectrum scaled
            according to the scaling input

    """
    if overlap_samples is None:
        overlap_samples = window_samples // 2
    step = window_samples - overlap_samples
    x_seg = np.lib.stride_tricks.sliding_window_view(
        x, window_shape=window_samples
    )[::step]  # (num_blocks, num_samples)

    if detrend:
        x_seg = scipy.signal.detrend(x_seg, type=detrend, axis=-1)

    win = scipy.signal.get_window(window, window_samples)
    x_seg = x_seg * win
    x_fft = np.fft.rfft(x_seg, n=window_samples, axis=-1)

    # Magnitude with coherent-gain correction
    spectrum = np.abs(x_fft)

    # One-sided amplitude scaling: double only non-DC/non-Nyquist bins
    if window_samples % 2 == 0:
        spectrum[..., 1:-1] *= 2.0
    else:
        spectrum[..., 1:] *= 2.0

    spectrum /= win.sum()  # coherent gain
    frequencies = np.fft.rfftfreq(window_samples, 1/fs)

    # Obtain RMS values of spectrum
    rms_vals = np.sqrt(np.mean(np.square(spectrum), axis=1))
    median_rms = np.median(rms_vals)
    accepted_idc = np.where(rms_vals < artifact_rejection_thr*median_rms)[0]

    if len(accepted_idc):
        spec_avg = spectrum[accepted_idc].mean(axis=0)
    else:
        spec_avg = np.zeros(len(spectrum[0]))

    return frequencies.astype(np.float32), spec_avg.astype(np.float32)


def process_spectrum(
    recorded_signal: npt.NDArray[np.float32],
    fs: float,
    window_samples: int,
    correction_tf: MicroTransferFunction | None,
    artifact_rejection_thr: float = 1.8
) -> npt.NDArray[np.float32]:
    """Processes recorded signal and obtains spectrum from averaged data.

    Args:
        recorded_signal: Float array of measurement data
        fs: Sampling frequency the signal was recorded with
        window_samples: Number of samples for each window for
          asynchronous spectrum estimations
        correction_tf: Transfer function of the microphone
        artifact_rejection_thr: Threshold representing a factor where blocks
          larger than factor*mean_rms are rejected

    Returns:
        Float array containing the asynchronous averaged spectrum
    """

    spectrum = None

    if len(recorded_signal) > window_samples:
        _, spectrum = welch_artifact_rejection(
            recorded_signal,
            fs,
            'hann',
            window_samples,
            artifact_rejection_thr=artifact_rejection_thr
        )
        if np.max(spectrum) == 0:
            spectrum[:] = np.finfo(np.float32).eps
        if correction_tf is None:
            spectrum = 20 * np.log10(spectrum)
        else:
            spectrum /= correction_tf.amplitudes
            spectrum = 20 * np.log10(spectrum/20)

    else:
        spectrum = np.abs(np.fft.rfft(np.zeros(window_samples, np.float32)))
    return spectrum


def get_results(
    sync_msrmt: SyncMsrmt,
    info: SoaeUpdateInfo
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Processes data and returns plot results.

    If the measurement is currently running, the recorded signal is obtained
    and a asynchronously averaged spectrum is estimated.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos about the measurement.

    Returns:
        tuple[recorded_signal, spectrum]

        - **recorded_signal**: Float array with recorded signal
        - **spectrum**: Float array with estimated spectrum

    """

    if sync_msrmt.state in [
        MsrmtState.RECORDING,
        MsrmtState.END_RECORDING,
        MsrmtState.FINISHING,
        MsrmtState.FINISHED
    ]:

        spectrum = None
        recorded_signal = sync_msrmt.get_recorded_signal()

        spectrum = process_spectrum(
            recorded_signal,
            info.fs,
            info.block_size,
            info.correction_tf,
            info.artifact_rejection_thr
        )

        return recorded_signal, spectrum

    return np.zeros(0,np.float32), np.zeros(0,np.float32)


def update_plot_data(
    recorded_signal: npt.NDArray[np.float32],
    spectrum: npt.NDArray[np.float32],
    info: SoaeUpdateInfo
) -> tuple[Line2D, Line2D]:
    """Updates the plot data.

    Creates and sets the plot objects for the data.

    Args:
        recorded_signal: Float array containing the raw signal.
        spectrum: Float array containing the spectral estimate.
        info: Info object containing meta infos and plot objects.

    Returns:
        tuple[line_time, line_spec]

        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal

    """

    if len(spectrum) == 0:
        return info.plot_info.line_time, info.plot_info.line_spec

    n_samples_displayed = int(
        info.plot_info.live_display_duration*1E-3*info.fs
    )

    if len(recorded_signal) < n_samples_displayed:
        return info.plot_info.line_time, info.plot_info.line_spec

    x_data = -np.flipud(np.arange(n_samples_displayed)/info.fs*1E3)

    info.plot_info.line_time.set_data(
        x_data,
        recorded_signal[-n_samples_displayed:]
    )

    # Update y-axis limits.
    spec_min = min(spectrum[1:])
    spec_max = max(spectrum)
    padding = 15  # dB of padding on top and bottom

    spec_min = min(spectrum[1:])
    spec_max = max(spectrum)
    padding = 15  # dB of padding on top and bottom
    info.plot_info.ax_spec.set_ylim(spec_min - padding, spec_max + padding)
    info.plot_info.line_spec.set_ydata(spectrum)

    # if (spec_min < info.plot_info.ax_spec.get_ylim()[0]

    return info.plot_info.line_time, info.plot_info.line_spec


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    info: SoaeUpdateInfo
) -> tuple[Line2D, Line2D]:
    """Processes results from data and updates the plots.

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
        return info.plot_info.line_time, info.plot_info.line_spec

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.set_state(MsrmtState.FINISHED)
        logger.info('Recording complete. Please close window to continue.')

    recorded_signal, spectrum = get_results(sync_msrmt, info)

    return update_plot_data(recorded_signal, spectrum, info)


def start_plot(sync_msrmt: SyncMsrmt, info: SoaeUpdateInfo) -> None:
    """Start the live plot."""
    _ = FuncAnimation(
        info.plot_info.fig,
        update_msrmt,
        fargs=(sync_msrmt, info,),
        interval=info.plot_info.update_interval,
        blit=True,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()
    if sync_msrmt.state is not MsrmtState.FINISHED:
        sync_msrmt.state = MsrmtState.CANCELED


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
    recorded_signal, spectrum = get_results(sync_msrmt, info)
    _, ax_time, line_time, ax_spec, line_spec = setup_plot(
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
    plt.tight_layout()
    plt.show()


class SoaeRecorder:
    """Class to manage an SOAE recording."""

    signals: list[PeriodicSignal]
    """List of output signals for each channel

    These are mute signals for SOAE acquisition.
    """

    update_info: SoaeUpdateInfo
    """Instance to control SOAE measurement updates."""

    msrmt: SyncMsrmt
    """Instance to perform a synchronized OAE measurement."""

    subject: str
    """Name/ID of the subject to be used for the measurement file name."""

    ear: str
    """Recording ear (left/right) to be used for the measurement file name."""

    logger: Logger
    """Class logger for debug, info, warning, and error messages."""

    def __init__(
        self,
        msrmt_params: SoaeMsrmtParams,
        subject: str = '',
        ear: str = '',
        log: Logger | None = None
    ) -> None:
        """Creates an SOAE recorder from measurement parameters."""
        self.logger = log or get_logger()
        self.subject = subject
        self.ear = ear
        num_block_samples = int(
            msrmt_params['block_duration'] * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            msrmt_params['num_averaging_blocks'] * num_block_samples
        )
        # block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = num_total_recording_samples / DeviceConfig.sample_rate

        self.signals = []
        self.generate_output_signals(num_total_recording_samples)

        soae_info = self.setup_info(recording_duration, num_block_samples)
        self.update_info = SoaeUpdateInfo(
            soae_info,
            DeviceConfig.sample_rate,
            num_block_samples,
            msrmt_params['artifact_rejection_thresh']
        )
        # Prepare measurement
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_block_samples,
            DeviceConfig.device_buffer_size
        )
        hw_data = HardwareData(
            2,
            2,
            DeviceConfig.input_device,
            DeviceConfig.output_device
        )
        self.msrmt = SyncMsrmt(rec_data, hw_data, self.signals)

    def record(self) -> None:
        """Starts the recording."""
        self.logger.info("Starting SOAE recording...")
        self.msrmt.start_msrmt(start_plot, self.update_info)

        # Plot offline results after measurement
        self.logger.info(
            'Showing offline results. Please close window to continue.'
        )
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
        parts = [
            'soae_msrmt',
            time_stamp,
            helpers.sanitize_filename_part(self.subject),
            helpers.sanitize_filename_part(self.ear),
        ]
        file_name = "_".join(filter(None, parts))
        save_path = os.path.join(save_path, file_name)
        recorded_signal, spectrum = get_results(self.msrmt, self.update_info)
        np.savez(
            save_path,
            spectrum=spectrum,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate
        )
        self.logger.info("Measurement saved to %s.", save_path)

    def generate_output_signals(self, num_total_samples: int) -> None:
        """Generates the output signals for playback."""
        # Generate output signals
        signal1 = PeriodicSignal(
            np.zeros(DeviceConfig.device_buffer_size, dtype=np.float32),
            num_total_samples
        )
        signal2 = PeriodicSignal(
            np.zeros(DeviceConfig.device_buffer_size, dtype=np.float32),
            num_total_samples
        )
        self.signals.append(signal1)
        self.signals.append(signal2)

    def setup_info(
        self,
        recording_duration: float,
        num_block_samples: int
    ) -> SoaePlotInfo:
        """Sets up live plot and measurement information."""

        update_time = (num_block_samples / DeviceConfig.sample_rate) * 1E3

        fig, ax_time, line_time, ax_spec, line_spec = setup_plot(
            recording_duration,
            DeviceConfig.sample_rate,
            num_block_samples,
            update_time
            #DeviceConfig.live_display_duration
        )
        return SoaePlotInfo(
            fig,
            ax_time,
            line_time,
            ax_spec,
            line_spec,
            update_time,
            update_time
            # DeviceConfig.update_interval,
            # DeviceConfig.live_display_duration
        )
