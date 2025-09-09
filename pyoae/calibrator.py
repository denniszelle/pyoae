"""Classes and functions to calibrate output channels.

This module is not intended to be run directly.
"""

from dataclasses import dataclass
from datetime import datetime
# import os

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt

from pyoae import files
from pyoae import generator
from pyoae.calib import MicroTransferFunction, SpeakerCalibData
from pyoae.device.device_config import DeviceConfig
from pyoae.protocols import CalibMsrmtParams
from pyoae.signals import Signal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


@dataclass
class PlotInfo:
    """Container with data for the measurement plot."""

    fig: Figure
    """Figure corresponding to the plot window."""

    ax_time: Axes
    """Axis object of the time plot."""

    line_time: Line2D
    """Line object for the time plot."""

    update_interval: float
    """Interval to apply processing and plot update during measurement."""

    live_display_duration: float
    """Duration to display time domain plot in ms."""


@dataclass
class UpdateInfo:
    """Container with data for calibration measurement updates."""

    plot_info: PlotInfo
    """Storage for the measurement plot."""

    fs: float
    """Sampling frequency in Hz."""

    block_size: int
    """Number of samples in each block."""

    input_trans_fun: MicroTransferFunction | None = None
    """Handle to microphone transfer function.

    A microphone transfer function is used to correct the
    recorded signal from the microphone characteristics.

    Note:
        This is a dummy object for future implementation.
    """


def setup_plot(
    recording_duration: float,
    fs: float
) -> tuple[Figure, Axes, Line2D]:
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Set up time plot
    x_wave = np.arange(round(recording_duration*fs), dtype=np.float32) / fs * 1E3
    y_wave = np.zeros_like(x_wave)
    line_time, = ax.plot(
        x_wave,
        y_wave
    )
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, recording_duration)
    ax.set_title("Recorded Waveform")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (full scale)")

    return fig, ax, line_time


def setup_offline_plot(
    frequency_range: tuple[float, float],
    is_calib_available:bool=False
) -> list[Axes]:
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
        list[axes]

        - **fig**: Object containing the plots
        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    _, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes: list[Axes]

    # Set up time plot
    for i, ax in enumerate(axes):
        ax.set_xlim(frequency_range[0], frequency_range[1])
        ax.set_ylim(-50, 100)
        ax.set_xscale('log')
        ax.set_title(f"Spectrum of Channel {i}")
        ax.set_xlabel("Frequency (Hz)")
        if is_calib_available:
            ax.set_ylabel('Level (dB SPL)')
        else:
            ax.set_ylabel("Level (dBFS)")
    return axes


def process_spectrum(
    recorded_signal: npt.NDArray[np.float32],
    correction_tf: MicroTransferFunction | None
) -> npt.NDArray[np.float32]:
    """Processes recorded signal to obtain spectrum.

    Args:
        recorded_signal: float array of measurement data
        block_size: Size of each recording block in samples
        correction_tf: Transfer function of the microphone

    Returns:
        Array of floats containing the spectrum.
    """
    spectrum = 2*np.abs(np.fft.rfft(recorded_signal))/len(recorded_signal)
    # dBFS and dB SPL represent RMS values
    # assume FFT bins represent sine waves and estimate
    # RMS by dividing by sqrt(2)
    spectrum /= np.sqrt(2)
    if correction_tf is None:
        spectrum = 20*np.log10(spectrum)
    else:
        spectrum /= correction_tf.amplitudes
        spectrum = 20*np.log10(spectrum/20)

    return spectrum.astype(np.float32)


def get_results(sync_msrmt: SyncMsrmt) -> npt.NDArray[np.float32]:
    """Processes data and returns plot results.

    If the measurement is currently running, the recorded signal
    is obtained and a synchronously averaged spectrum is estimated.

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
        MsrmtState.RECORDING,
        MsrmtState.END_RECORDING,
        MsrmtState.FINISHING,
        MsrmtState.FINISHED
    ]:
        recorded_signal = sync_msrmt.get_recorded_signal()
        return recorded_signal

    return np.zeros(0,np.float32)


def get_channel_spectra(
    sync_msrmt: SyncMsrmt,
    info: UpdateInfo
) -> list[npt.NDArray[np.float32]]:
    """Processes data and returns spectra per output channels.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos about the measurement.

    Returns:
        list[spectrum_ch01, spectrum_ch02, ...]
    """


    recorded_signal = sync_msrmt.get_recorded_signal()
    if len(recorded_signal) != sync_msrmt.recording_data.msrmt_samples:
        return []

    # TODO: make channel selection dynamic using properties

    block_size = int(0.5 * info.block_size)
    print(f'Block size {block_size} for transfer function.')

    spectrum_ch01 = process_spectrum(
        recorded_signal[:block_size],
        info.input_trans_fun,
    )
    spectrum_ch02 = process_spectrum(
        recorded_signal[block_size:],
        info.input_trans_fun
    )
    return [spectrum_ch01, spectrum_ch02]


def update_plot_data(
    recorded_signal: npt.NDArray[np.float32],
    info: UpdateInfo
) -> Line2D:
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

    n_samples_displayed = int(
        info.plot_info.live_display_duration * 1E-3 * info.fs
    )

    num_recorded_samples = len(recorded_signal)

    if num_recorded_samples < n_samples_displayed:
        return info.plot_info.line_time

    #x_data = -np.flipud(np.arange(n_samples_displayed) / info.fs * 1E3)
    x_data = np.arange(num_recorded_samples) / info.fs

    info.plot_info.line_time.set_data(
        x_data,
        recorded_signal
    )

    return info.plot_info.line_time


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    info: UpdateInfo
) -> tuple[Line2D,]:
    """Processes results from data and update the plots.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos and plot objects

    Returns:
        tuple[line_time, ]

        - **line_time**: Line object with the time-domain data of the signal
    Notes:
        A tuple is returned with at least one line object to be compatible
          with `FuncAnimation`.
    """

    del frame

    if sync_msrmt.state == MsrmtState.FINISHED:
        # plt.close(info.fig)
        return (info.plot_info.line_time, )

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.state = MsrmtState.FINISHED

    recorded_signal = get_results(sync_msrmt)

    return (update_plot_data(recorded_signal, info), )


def start_plot(sync_msrmt: SyncMsrmt, info: UpdateInfo) -> None:
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


def plot_offline(
    sync_msrmt: SyncMsrmt,
    info: UpdateInfo,
    mt_frequencies: npt.NDArray[np.float32],
    output_amplitude: float
) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.
    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return

    has_input_calib =  info.input_trans_fun is not None
    axes = setup_offline_plot((100, 20000), has_input_calib)

    ch_spectra = get_channel_spectra(sync_msrmt, info)

    # plot channel spectra
    num_bins = len(ch_spectra[0])
    df = DeviceConfig.sample_rate / (0.5*info.block_size)
    f = np.arange(num_bins, dtype=np.float32) * df

    # find frequency indices to sample transfer function
    mt_bin_idx = np.round((mt_frequencies/df)).astype(np.int_)

    # calculate the remaining output capacity to
    # retrieve the maximum output level
    output_overhead = 20*np.log10(1/output_amplitude)

    padding = 15  # dB of padding on top and bottom
    for i, ax in enumerate(axes):
        out_db_spl = ch_spectra[i][mt_bin_idx]

        p_out_peak = np.sqrt(2) * 20 * 10**(out_db_spl/20)
        p_out_max = p_out_peak / output_amplitude

        out_max_db_spl = 20*np.log10(p_out_max/(np.sqrt(2)*20))

        spec_min = min(ch_spectra[i])
        spec_max = max(ch_spectra[i] + output_overhead)

        ax.plot(f, ch_spectra[i])
        ax.plot(mt_frequencies, out_db_spl, 'rx')
        ax.plot(mt_frequencies, out_db_spl + output_overhead, 'bo-')
        ax.plot(mt_frequencies, out_max_db_spl, 'r--')
        ax.set_ylim(spec_min - padding, spec_max + padding)

    plt.tight_layout()
    plt.show()


class OutputCalibRecorder:
    """Class to manage a DPOAE recording."""

    # stimulus: MultiToneStimulus
    # """Parameters of multi tone."""

    mt_frequencies: npt.NDArray[np.float32]

    output_amplitude: float

    signals: list[Signal]
    """List of output signals for each channel."""

    update_info: UpdateInfo
    """Instance to control measurement updates."""

    msrmt: SyncMsrmt
    """Instance to perform a synchronized measurement."""

    results: SpeakerCalibData | None
    """Calibration results for output channels."""

    def __init__(
        self,
        msrmt_params: CalibMsrmtParams,
        mic_trans_fun: MicroTransferFunction | None = None
    ) -> None:
        """Creates a DPOAE recorder for given measurement parameters."""
        num_block_samples = int(
            msrmt_params['block_duration'] * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            msrmt_params['num_channels'] * num_block_samples
        )
        block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = num_total_recording_samples / DeviceConfig.sample_rate

        if mic_trans_fun:
            mic_trans_fun.num_samples = num_block_samples
            mic_trans_fun.sample_rate = DeviceConfig.sample_rate
            mic_trans_fun.interpolate_transfer_fun()

        if block_duration != msrmt_params["block_duration"]:
            print(
                f'Block duration adjusted to {block_duration*1E3:.2f} ms'
            )

        self.signals = []
        self.generate_output_signals(
            msrmt_params,
            num_block_samples,
            num_total_recording_samples
        )
        msrmt_info = self.setup_info(recording_duration)
        self.update_info = UpdateInfo(
            msrmt_info,
            DeviceConfig.sample_rate,
            num_total_recording_samples,
            input_trans_fun=mic_trans_fun
        )
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_total_recording_samples,
            DeviceConfig.device_buffer_size
        )
        hw_data = HardwareData(
            2,
            2,
            DeviceConfig.input_device,
            DeviceConfig.output_device,
        )
        self.msrmt = SyncMsrmt(
            rec_data,
            hw_data,
            self.signals
        )
        self.results = None

    def record(self) -> None:
        """Starts the calibration."""
        print("Starting calibration...")
        # `start_msrmt` starts the application loop
        self.msrmt.start_msrmt(start_plot, self.update_info)

        # Compute calibration results
        self.compute_calib_results()

        if self.results is None:
            return

        # Plot all data and final result after user has
        # closed the live-measurement window.
        plot_offline(
            self.msrmt,
            self.update_info,
            self.mt_frequencies,
            self.output_amplitude
        )

    def compute_calib_results(self) -> None:
        """Computes the output-channel transfer functions."""
        if self.msrmt.state != MsrmtState.FINISHED:
            return

        ch_spectra = get_channel_spectra(self.msrmt, self.update_info)

        # plot channel spectra
        df = DeviceConfig.sample_rate / (0.5*self.update_info.block_size)

        # find frequency indices to sample transfer function
        mt_bin_idx = np.round((self.mt_frequencies/df)).astype(np.int_)

        calib_results: list[npt.NDArray[np.float32]] = []
        for ch_spec in ch_spectra:
            out_db_spl = ch_spec[mt_bin_idx]

            p_out_peak = np.sqrt(2) * 20 * 10**(out_db_spl/20)
            p_out_max = p_out_peak / self.output_amplitude
            calib_results.append(p_out_max.astype(np.float32))

        cur_time = datetime.now()
        time_stamp = cur_time.strftime("%y%m%d-%H%M%S")
        self.results = {
            "date": time_stamp,
            "frequencies": self.mt_frequencies.tolist(),
            "max_out_ch01": calib_results[0].tolist(),
            "max_out_ch02": calib_results[0].tolist(),
            "phase_ch01": np.zeros_like(self.mt_frequencies).tolist(),
            "phase_ch02": np.zeros_like(self.mt_frequencies).tolist(),
        }

    def save_recording(self) -> None:
        """Stores the measurement data in binary file."""
        if self.results is not None:
            files.save_output_calibration(self.results)

    def generate_output_signals(
        self,
        msrmt_params: CalibMsrmtParams,
        num_block_samples: int,
        num_total_recording_samples: int,
    ) -> None:
        """Generates the output signals for playback."""
        self.mt_frequencies = generator.compute_mt_frequencies(
            msrmt_params["f_start"],
            msrmt_params["f_stop"],
            msrmt_params["lines_per_octave"]
        )
        num_mt_frequencies = len(self.mt_frequencies)
        mt_phases = generator.compute_mt_phases(num_mt_frequencies)

        mt_signal = generator.generate_mt_signal(
            num_block_samples,
            DeviceConfig.sample_rate,
            self.mt_frequencies,
            mt_phases
        )
        self.output_amplitude = msrmt_params["amplitude_per_line"]
        mt_signal *= self.output_amplitude

        # we always use rising and falling edges
        ramp_len = int(
            DeviceConfig.ramp_duration * 1E-3 * DeviceConfig.sample_rate
        )
        ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
        ramp = ramp.astype(np.float32)

        # apply ramps
        mt_signal[:ramp_len] *= ramp
        mt_signal[-ramp_len:] *= ramp[::-1]

        # create channel-specific stimuli by zero-padding
        stimulus1 = np.r_[mt_signal, np.zeros_like(mt_signal)]
        stimulus2 = np.r_[np.zeros_like(mt_signal), mt_signal]

        signal1 = Signal(stimulus1, num_total_recording_samples)
        signal2 = Signal(stimulus2, num_total_recording_samples)

        self.signals.append(signal1)
        self.signals.append(signal2)

    def setup_info(
        self,
        recording_duration: float
    ) -> PlotInfo:
        """Sets up live plot and measurement information."""
        fig, ax_time, line_time = setup_plot(
            recording_duration,
            DeviceConfig.sample_rate
        )
        return PlotInfo(
            fig,
            ax_time,
            line_time,
            DeviceConfig.update_interval,
            DeviceConfig.live_display_duration
        )
