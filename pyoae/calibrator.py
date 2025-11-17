"""Classes and functions to calibrate output channels.

This module is not intended to be run directly.
"""

from datetime import datetime
from logging import Logger

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt

from pyoae import files
from pyoae import generator
from pyoae import get_logger
from pyoae.calib import (
    MicroTransferFunction,
    OutputCalibration,
    SpeakerCalibData
)
from pyoae.device.device_config import DeviceConfig
from pyoae.msrmt_context import MsrmtContext
from pyoae.plot_context import PlotContext
from pyoae.protocols import CalibMsrmtParams
from pyoae.signals import Signal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


logger = get_logger(__name__)


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
    x_wave = np.arange(round(recording_duration*fs), dtype=np.float32) / fs
    y_wave = np.zeros_like(x_wave)
    line_time, = ax.plot(
        x_wave,
        y_wave
    )
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, recording_duration)
    ax.set_title("Recorded Waveform")
    ax.set_xlabel("Time (s)")
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
    msrmt_ctx: MsrmtContext
) -> list[npt.NDArray[np.float32]]:
    """Processes data and returns spectra per output channels.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.

    Returns:
        list[spectrum_ch01, spectrum_ch02, ...]
    """


    recorded_signal = sync_msrmt.get_recorded_signal()
    if len(recorded_signal) != sync_msrmt.recording_data.msrmt_samples:
        return []

    # TODO: make channel selection dynamic using properties

    block_size = int(0.5 * msrmt_ctx.block_size)
    logger.debug('Block size for transfer function: %d.', block_size)

    spectrum_ch01 = process_spectrum(
        recorded_signal[:block_size],
        msrmt_ctx.input_trans_fun,
    )
    spectrum_ch02 = process_spectrum(
        recorded_signal[block_size:],
        msrmt_ctx.input_trans_fun
    )
    return [spectrum_ch01, spectrum_ch02]


def update_plot_data(
    recorded_signal: npt.NDArray[np.float32],
    msrmt_ctx: MsrmtContext,
    plot_ctx: PlotContext
) -> Line2D:
    """Updates the plot data.

    Create and set the plot objects for the data.

    Args:
        recorded_signal: Float array containing the raw signal.
        spectrum: Float array containing the spectral estimate.
        msrmt_ctx: Parameters and instances to control the measurement.
        plot_ctx: Parameters and instances to control plots.

    Returns:
        tuple[line_time, line_spec]

        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    n_samples_displayed = int(
        plot_ctx.live_display_duration * 1E-3 * msrmt_ctx.fs
    )

    num_recorded_samples = len(recorded_signal)

    if num_recorded_samples < n_samples_displayed:
        return plot_ctx.line

    #x_data = -np.flipud(np.arange(n_samples_displayed) / info.fs * 1E3)
    x_data = np.arange(num_recorded_samples) / msrmt_ctx.fs

    plot_ctx.line.set_data(
        x_data,
        recorded_signal
    )

    return plot_ctx.line


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    plot_ctx: PlotContext
) -> tuple[Line2D,]:
    """Processes results from data and update the plots.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.
        plot_ctx: Parameters and instances to control plots.

    Returns:
        tuple[line_time, ]

        - **line_time**: Line object with the time-domain data of the signal
    Notes:
        A tuple is returned with at least one line object to be compatible
          with `MsrmtFuncAnimation`.
    """

    del frame

    if sync_msrmt.state == MsrmtState.FINISHED:
        if msrmt_ctx.msrmt_anim is not None and msrmt_ctx.non_interactive:
            msrmt_ctx.msrmt_anim.stop_animation()
        return (plot_ctx.line, )

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.set_state(MsrmtState.FINISHED)
        if msrmt_ctx.non_interactive:
            logger.info('Recording complete.')
        else:
            logger.info('Recording complete. Please close window to continue.')

    recorded_signal = get_results(sync_msrmt)

    return (update_plot_data(recorded_signal, msrmt_ctx, plot_ctx), )


def plot_offline(
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    mt_frequencies: npt.NDArray[np.float32],
    output_amplitude: float
) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.
    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return

    has_input_calib =  msrmt_ctx.input_trans_fun is not None
    axes = setup_offline_plot((100, 20000), has_input_calib)

    ch_spectra = get_channel_spectra(sync_msrmt, msrmt_ctx)

    # plot channel spectra
    num_bins = len(ch_spectra[0])
    df = DeviceConfig.sample_rate / (0.5*msrmt_ctx.block_size)
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


def plot_result_file(results: OutputCalibration) -> None:
    """Plots output calibration from result file."""
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.set_xlim(100, 20000)
    ax.set_ylim(0, 120)
    ax.set_xscale('log')
    ax.set_title(f"Calibration {results.date} - Maximum Output Level")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel('Level (dB SPL)')

    line_styles = ['rx-', 'b.-']
    for i in range(2):
        p_out_max = results.amplitudes[i,:]
        out_max_db_spl = 20*np.log10(p_out_max/(np.sqrt(2)*20))
        y_min = 0
        y_max = np.ceil(max(out_max_db_spl)/20)*20
        ax.plot(results.frequencies, out_max_db_spl, line_styles[i])
        ax.set_ylim(y_min, y_max)
    ax.legend(['Ch. 0', 'Ch. 1'])
    ax.grid(True, which='both')
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

    plot_ctx: PlotContext
    """Instance to plot context for measurement visualization."""

    msrmt_ctx: MsrmtContext
    """Instance to perform a synchronized OAE measurement."""

    msrmt: SyncMsrmt
    """Instance to perform a synchronized measurement."""

    results: SpeakerCalibData | None
    """Calibration results for output channels."""

    logger: Logger
    """Class logger for debug, info, warning, and error messages."""

    def __init__(
        self,
        msrmt_params: CalibMsrmtParams,
        mic_trans_fun: MicroTransferFunction | None = None,
        log: Logger | None = None
    ) -> None:
        """Creates a simple multi-tone output calibrator."""
        self.logger = log or get_logger()
        num_block_samples = int(
            msrmt_params['block_duration'] * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            msrmt_params['num_channels'] * num_block_samples
        )
        block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = (
            num_total_recording_samples / DeviceConfig.sample_rate
        )

        if mic_trans_fun:
            mic_trans_fun.num_samples = num_block_samples
            mic_trans_fun.sample_rate = DeviceConfig.sample_rate
            mic_trans_fun.interpolate_transfer_fun()

        if block_duration != msrmt_params["block_duration"]:
            self.logger.warning(
                'Block duration adjusted to {%.2f} ms.',
                block_duration * 1E3
            )

        self.signals = []
        self.generate_output_signals(
            msrmt_params,
            num_block_samples,
            num_total_recording_samples
        )
        self.plot_ctx = self.setup_plot_context(recording_duration)
        self.msrmt_ctx = MsrmtContext(
            fs=DeviceConfig.sample_rate,
            block_size=num_total_recording_samples,
            input_trans_fun=mic_trans_fun,
            artifact_rejection_thr=-1.0,  # not used
            non_interactive=False,
            msrmt_anim=None
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
        self.logger.info("Starting output calibration...")
        # `start_msrmt` starts the application loop
        self.msrmt.start_msrmt(update_msrmt, self.msrmt_ctx, self.plot_ctx)

        # Compute calibration results
        self.compute_calib_results()

        if self.results is None:
            return

        if not self.msrmt_ctx.non_interactive:
            # Plot all data and final result after user has
            # closed the live-measurement window.
            self.logger.info(
                'Showing offline results. Please close window to continue.'
            )
            plot_offline(
                self.msrmt,
                self.msrmt_ctx,
                self.mt_frequencies,
                self.output_amplitude
            )

    def compute_calib_results(self) -> None:
        """Computes the output-channel transfer functions."""
        if self.msrmt.state != MsrmtState.FINISHED:
            return

        ch_spectra = get_channel_spectra(self.msrmt, self.msrmt_ctx)

        # plot channel spectra
        df = DeviceConfig.sample_rate / (0.5*self.msrmt_ctx.block_size)

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
            "max_out_ch02": calib_results[1].tolist(),
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

        max_amplitude = np.max(mt_signal)
        if max_amplitude > DeviceConfig.max_digital_output:
            self.logger.warning(
                'Maximum output %.2f limited to maximum %.2f re FS.',
                max_amplitude,
                DeviceConfig.max_digital_output
            )
            self.logger.warning(
                'Output calibration results might be invalid.'
            )

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

    def setup_plot_context(
        self,
        recording_duration: float
    ) -> PlotContext:
        """Sets up live plot and measurement information."""
        fig, ax_time, line_time = setup_plot(
            recording_duration,
            DeviceConfig.sample_rate
        )
        return PlotContext(
            fig=fig,
            axes=ax_time,
            line=line_time,
            update_interval=DeviceConfig.update_interval,
            live_display_duration=DeviceConfig.live_display_duration
        )
