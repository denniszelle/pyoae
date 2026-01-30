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

from datetime import datetime
from logging import Logger
import os
from typing import Literal

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import scipy.signal

from pyoae import get_logger
from pyoae import helpers
from pyoae.calib import MicroTransferFunction
from pyoae.dsp.opt_avg import OptAverage
from pyoae.device.device_config import DeviceConfig
from pyoae.msrmt_context import MsrmtContext
from pyoae.protocols import MsrmtParams
from pyoae.signals import PeriodicSignal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


logger = get_logger()


SPECTRAL_PLOT_PADDING: float = 15.0
"""Padding for y-limits of spectral plot in dB."""


def setup_plot(
    recording_duration: float,
    fs: float,
    window_size: int,
    is_calib_available:bool=False
) -> tuple[Axes, Line2D, Axes, Line2D]:
    """Sets up the plots for RMS spectrum results.

    Top plot: shows the recorded time signal.
    Bottom plot: shows the estimated RMS spectrum.

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
    _, axes = plt.subplots(2, 1, figsize=(10, 6))
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

    return ax_time, line_time, ax_spec, line_spec


def average_spectrum(
    x: npt.NDArray[np.float32],
    fs: float,
    window: str = 'hann',
    window_samples: int = 256,
    overlap_samples: int | None = None,
    detrend: Literal['linear', 'constant'] = 'constant'
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Estimates a welch spectrum with optimized averaging.

    Args:
        x: Float array as time-domain signal
        fs: Sampling frequency of the measurement
        window: Type of the window (see scipy.signal.get_window)
        window_samples: Samples for each window
        overlap_samples: Window overlap samples, window_samples/2 by default
        detrend: Remove linear or constant trend beforehand

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

    spectrum /= np.sum(win)  # coherent gain
    frequencies = np.fft.rfftfreq(window_samples, 1/fs)

    # Obtain RMS values of spectrum
    rms_vals = np.sqrt(np.mean(np.square(spectrum), axis=1))

    # Apply optimized averaging
    opt_averager = OptAverage()
    opt_averager.setup(len(rms_vals))
    for i, rms_val_i in enumerate(rms_vals):
        opt_averager.set_noise_value(i, rms_val_i)
    opt_averager.evaluate_averaging()

    accepted_idc = opt_averager.accepted_idx

    if len(accepted_idc):
        spec_avg = spectrum[accepted_idc].mean(axis=0)
    else:
        spec_avg = np.zeros(len(spectrum[0]))

    return frequencies.astype(np.float32), spec_avg.astype(np.float32)


class SoaeRecorder:
    """Class to manage an SOAE recording."""

    signals: list[PeriodicSignal]
    """List of output signals for each channel

    These are mute signals for SOAE acquisition.
    """

    msrmt_ctx: MsrmtContext
    """Instance to context to control SOAE measurement updates."""

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
        msrmt_params: MsrmtParams,
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

        self.msrmt_ctx = MsrmtContext(
            fs=DeviceConfig.sample_rate,
            block_size=num_block_samples,
            input_trans_fun=None,
            non_interactive=False
        )
        # Prepare measurement
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_block_samples,
            DeviceConfig.device_buffer_size
        )
        # TODO: fix number of arguments to support bilateral PyOAE interface
        hw_data = HardwareData(
            2,
            2,
            DeviceConfig.input_device,
            DeviceConfig.output_device
        )
        # TODO: fix number of arguments to support bilateral PyOAE interface
        self.msrmt = SyncMsrmt(
            rec_data,
            hw_data,
            self.signals,
            msrmt_params['block_duration']
        )

    def record(self) -> None:
        """Starts the recording."""
        self.logger.info("Starting SOAE recording...")

        self.msrmt.run_msrmt()

        if not self.msrmt_ctx.non_interactive:
            self.logger.info(
                'Showing offline results. Please close window to continue.'
            )
            self.plot_offline()

    def _plot_offline(
        self,
        recorded_signal: npt.NDArray[np.float32],
        spectrum: npt.NDArray[np.float32]
    ) -> None:
        """Helper to plot the final results in a non-updating plot.

        This method obtains the results from the measurement object,
        creates a plot and shows the complete measurement
        as well as the RMS spectrum.

        Args:
            recorded_signal: recorded time signal
            spectrum: RMS-averaged spectrum of recorded signal
        """
        ax_time, line_time, ax_spec, line_spec = setup_plot(
            self.msrmt.recording_data.msrmt_duration,
            self.msrmt.recording_data.fs,
            self.msrmt_ctx.block_size,
            self.msrmt_ctx.input_trans_fun is not None
        )
        line_time.set_xdata(np.arange(len(recorded_signal))/self.msrmt_ctx.fs)
        line_time.set_ydata(recorded_signal)
        ax_time.set_xlim(0, self.msrmt.recording_data.msrmt_duration)
        ax_time.set_xlabel("Recording Time (s)")

        spec_min = np.floor((min(spectrum[1:]) / 5)) * 5
        spec_max = np.ceil(max(spectrum) / 5) * 5
        # set y limits with padding
        ax_spec.set_ylim(
            spec_min - SPECTRAL_PLOT_PADDING,
            spec_max + SPECTRAL_PLOT_PADDING
        )
        line_spec.set_ydata(spectrum)

    def plot_offline(self) -> None:
        """Shows the final results in a polished plot."""
        if self.msrmt.state != MsrmtState.FINISHED:
            return
        recorded_signal, spectrum = self.get_results()
        self._plot_offline(recorded_signal, spectrum)
        plt.tight_layout()
        plt.show()

    def get_results(
        self
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Processes data and returns plot results.

        If the measurement is currently running, the recorded signal is obtained
        and a asynchronously averaged spectrum is estimated.

        Returns:
            tuple[recorded_signal, spectrum]

            - **recorded_signal**: Float array with recorded signal
            - **spectrum**: Float array with estimated spectrum

        """

        if self.msrmt.state in [
            MsrmtState.RECORDING,
            MsrmtState.END_RECORDING,
            MsrmtState.FINISHING,
            MsrmtState.FINISHED
        ]:

            spectrum = None
            # TODO: add input channel to retrieve recorded signal
            recorded_signal = self.msrmt.get_recorded_signal()
            spectrum = self.process_spectrum(recorded_signal)

            return recorded_signal, spectrum

        return np.zeros(0,np.float32), np.zeros(0,np.float32)

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
        recorded_signal, spectrum = self.get_results()
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

    def process_spectrum(
        self,
        recorded_signal: npt.NDArray[np.float32],
        window: str = 'hann'
    ) -> npt.NDArray[np.float32]:
        """Processes recorded signal and obtains spectrum from averaged data.

        Args:
            recorded_signal: Float array of measurement data

        Returns:
            Float array containing the asynchronous averaged spectrum
        """

        correction_tf = None
        if self.msrmt_ctx.input_trans_fun is not None:
            # TODO: retrieve and use input channel
            input_ch = 0
            # TODO: retrieve correct micro TF from list
            correction_tf = self.msrmt_ctx.input_trans_fun[input_ch]

        spectrum = None

        if len(recorded_signal) > self.msrmt_ctx.block_size:
            _, spectrum = average_spectrum(
                recorded_signal,
                self.msrmt_ctx.fs,
                window,
                self.msrmt_ctx.block_size,
            )
            if np.max(spectrum) == 0:
                # pylint: disable=no-member
                spectrum[:] = np.finfo(np.float32).eps
                # pylint: enable=no-member
            if correction_tf is None:
                spectrum = 20 * np.log10(spectrum)
            else:
                spectrum /= correction_tf.amplitudes
                spectrum = 20 * np.log10(spectrum/20)

        else:
            spectrum = np.abs(np.fft.rfft(np.zeros(self.msrmt_ctx.block_size, np.float32)))

        return spectrum
