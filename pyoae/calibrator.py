"""Classes and functions to calibrate output channels.

This module is not intended to be run directly.
"""

from collections import Counter
from datetime import datetime
from logging import Logger

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
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
from pyoae.protocols import CalibMsrmtParams
from pyoae.signals import Signal
from pyoae.sync import (
    get_input_channels,
    HardwareData,
    RecordingData,
    SyncMsrmt,
    MsrmtState
)

logger = get_logger(__name__)


def setup_offline_plot(
    frequency_range: tuple[float, float],
    output_channels: list[int],
    input_channels: list[int],
    is_calib_available:bool=False,
) -> list[list[Axes]]:
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

    counter = Counter(input_channels)
    rows = max(counter.values()) + 1
    cols = len(counter)

    # Type ignore to set known dimensions of 2
    _, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 8),
        sharex='all',
        squeeze=False
    ) # type: ignore
    axes: list[list[Axes]]

    sorted_input_channels = list(counter.keys())

    for i, ax_i in enumerate(axes):
        for j, ax_ij in enumerate(ax_i):
            if i == len(axes) - 1:
                axes[-1][j].set_title(f'Channel Comparison for Input Channel {sorted_input_channels[j]}')
                axes[-1][j].set_xlabel("Frequency (Hz)")
            else:
                output_idc = np.where(np.asarray(input_channels) == sorted_input_channels[j])[0]
                if len(output_idc) > i:
                    output_channel_ij = output_channels[output_idc[i]]
                    ax_ij.set_xlim(frequency_range[0], frequency_range[1])
                    ax_ij.set_ylim(-50, 100)
                    ax_ij.set_xscale('log')
                    ax_ij.set_title(f"Spectrum of Output Channel {output_channel_ij}")
                    if is_calib_available:
                        ax_ij.set_ylabel('Level (dB SPL)')
                    else:
                        ax_ij.set_ylabel("Level (dBFS)")
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

    spectra = []

    block_size = int(
        msrmt_ctx.block_size/len(sync_msrmt.hardware_data.output_channels)
    )

    for i, _ in enumerate(sync_msrmt.hardware_data.output_channels):

        input_channel = sync_msrmt.hardware_data.input_channels[i]
        recorded_signal = sync_msrmt.get_recorded_signal(
            input_channel
        )
        if input_channel not in sync_msrmt.hardware_data.input_channels:
            logger.error('Invalid input channel')
            spectra.append(np.zeros(0, dtype=np.float32))
            continue
        input_channel_idx = (
            sync_msrmt.hardware_data.get_unique_input_channels().index(
                input_channel
            )
        )
        if msrmt_ctx.input_trans_fun is None:
            spectrum = process_spectrum(
                recorded_signal[i*block_size:(i+1)*block_size], None
            )
        else:
            spectrum = process_spectrum(
                recorded_signal[i*block_size:(i+1)*block_size],
                msrmt_ctx.input_trans_fun[input_channel_idx]
            )
        spectra.append(spectrum)
    return spectra


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
    f_min = np.floor((mt_frequencies.min() - 20) / 20) * 20
    f_max = np.ceil((mt_frequencies.max() + 500)/ 1000) * 1000
    f_min = max(20, f_min)
    axes = setup_offline_plot(
        (f_min, f_max),
        sync_msrmt.hardware_data.output_channels,
        sync_msrmt.hardware_data.input_channels,
        has_input_calib
    )

    ch_spectra = get_channel_spectra(sync_msrmt, msrmt_ctx)

    # plot channel spectra
    num_bins = len(ch_spectra[0])
    df = (
        DeviceConfig.sample_rate
        / (msrmt_ctx.block_size
        / len(sync_msrmt.hardware_data.output_channels))
    )
    f = np.arange(num_bins, dtype=np.float32) * df

    # find frequency indices to sample transfer function
    mt_bin_idx = np.round((mt_frequencies/df)).astype(np.int_)

    # calculate the remaining output capacity to
    # retrieve the maximum output level
    output_overhead = 20*np.log10(1/output_amplitude)

    padding = 15  # dB of padding on top and bottom
    line_vecs = ['b-', 'r-', 'g-']
    ax_cmp_min = 80
    ax_cmp_max = 120

    input_channels = sync_msrmt.hardware_data.input_channels
    counter = Counter(input_channels)
    sorted_input_channels = list(counter.keys())

    for i, ax_i in enumerate(axes):
        for j, ax_ij in enumerate(ax_i):
            if i < len(axes) - 1:

                output_idc = np.where(np.asarray(input_channels) == sorted_input_channels[j])[0]
                if len(output_idc) > i:
                    output_idx = output_idc[i]
                else:
                    continue
                # Plot measurement of channel ij
                out_db_spl = ch_spectra[output_idx][mt_bin_idx]
                p_out_peak = np.sqrt(2) * 20 * 10**(out_db_spl/20)
                p_out_max = p_out_peak / output_amplitude
                out_max_db_spl = 20*np.log10(p_out_max/(np.sqrt(2)*20))

                spec_min = min(ch_spectra[output_idx])
                spec_max = max(ch_spectra[output_idx] + output_overhead)

                ax_ij.plot(f, ch_spectra[output_idx], linewidth=0.5, color='k')
                ax_ij.plot(mt_frequencies, out_db_spl, 'ro', markersize=2)

                ax_ij.plot(
                    mt_frequencies,
                    out_db_spl + output_overhead,
                    'ko',
                    markersize=3,
                    markerfacecolor='w'
                )
                if len(line_vecs) > i:
                    axes[-1][j].plot(mt_frequencies, out_max_db_spl, line_vecs[i])
                    ax_ij.plot(mt_frequencies, out_max_db_spl, line_vecs[i])
                else:
                    axes[-1][j].plot(mt_frequencies, out_max_db_spl)
                    ax_ij.plot(mt_frequencies, out_max_db_spl)
                ax_ij.set_ylim(spec_min - padding, spec_max + padding)
                axes[-1][j].grid(True, which='both')
                out_lower_bnd = np.floor(out_max_db_spl.min()/20)*20
                out_upper_bnd = np.ceil(out_max_db_spl.max()/20)*20
                ax_cmp_min = min(ax_cmp_min, out_lower_bnd)
                ax_cmp_max = max(ax_cmp_max, out_upper_bnd)
                axes[-1][j].set_ylim(ax_cmp_min, ax_cmp_max)

    plt.tight_layout()
    plt.show()


def plot_result_file(results: OutputCalibration) -> None:
    """Plots output calibration from result file."""

    counter = Counter(results.input_channels)
    cols = len(counter)

    # Type ignore to set known dimensions of 2
    fig, axes = plt.subplots(
        1,
        cols,
        figsize=(12, 6),
        sharex='all',
        squeeze=False
    ) # type: ignore
    axes: list[list[Axes]]

    sorted_input_channels = list(counter.keys())

    line_styles = ['b.-', 'rx-', 'gd-']

    f_min = np.floor((results.frequencies.min() - 20) / 20) * 20
    f_max = np.ceil((results.frequencies.max() + 500)/ 1000) * 1000
    f_min = max(20, f_min)

    for i, input_channel_i in enumerate(sorted_input_channels):

        y_min = None
        y_max = None

        ax_i = axes[0][i]
        ax_i.set_xlim(f_min, f_max)
        ax_i.set_xscale('log')
        ax_i.set_title(f"Maximum Output Level - Mic Channel {input_channel_i}")
        ax_i.set_xlabel("Frequency (Hz)")
        ax_i.set_ylabel('Level (dB SPL)')

        output_idc = np.where(
            np.asarray(results.input_channels) == input_channel_i
        )[0]
        for j, output_idx_j in enumerate(output_idc):
            output_channel_j = results.output_channels[output_idx_j]

            p_out_max = results.amplitudes[output_idx_j,:]
            out_max_db_spl = 20*np.log10(p_out_max/(np.sqrt(2)*20))
            y_max = np.ceil(max(out_max_db_spl)/20)*20
            if y_max <= 0:
                y_min = y_max-100
            else:
                y_min = 0
            if i < len(line_styles):
                ax_i.plot(
                    results.frequencies,
                    out_max_db_spl,
                    line_styles[j],
                    label=f'Channel {output_channel_j}'
                )
            else:
                ax_i.plot(
                    results.frequencies,
                    out_max_db_spl,
                    label=f'Channel {output_channel_j}'
                )
        if y_min is not None and y_max is not None:
            ax_i.set_ylim(y_min, y_max)
        ax_i.legend()
        ax_i.grid(True, which='both')
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f'Calibration {results.date}')
    plt.tight_layout()
    plt.show()


class OutputCalibRecorder:
    """Class to manage a DPOAE recording."""

    mt_frequencies: npt.NDArray[np.float32]

    output_amplitude: float

    signals: list[Signal]
    """List of output signals for each channel."""

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
        output_channels: list[int],
        mic_trans_fun: list[MicroTransferFunction] | None = None,
        log: Logger | None = None
    ) -> None:
        """Creates a simple multi-tone output calibrator."""

        self.logger = log or get_logger()
        num_block_samples = int(
            msrmt_params['block_duration'] * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            len(output_channels) * num_block_samples
        )
        block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = (
            num_total_recording_samples / DeviceConfig.sample_rate
        )

        # Set to false if major problem occured during calibration
        self.ready_to_record = True
        self.results = None

        if block_duration != msrmt_params['block_duration']:
            self.logger.warning(
                'Block duration adjusted to {%.2f} ms.',
                block_duration * 1E3
            )

        # Setup hardware data
        active_in_channels = list({
            b for a, b in DeviceConfig.output_input_mapping
            if a in output_channels
        })

        n_in_channels = max(
            *active_in_channels, DeviceConfig.sync_channels[1]
        )+1
        n_out_channels = max(output_channels) + 1
        hw_data = HardwareData(
            n_in_channels,
            n_out_channels,
            DeviceConfig.input_device,
            DeviceConfig.output_device,
            output_channels,
            get_input_channels(output_channels)
        )

        if mic_trans_fun:
            mic_transfer_functions = []
            if len(mic_trans_fun) == len(active_in_channels):
                for trans_fun_i in mic_trans_fun:
                    trans_fun_i.num_samples = num_block_samples
                    trans_fun_i.sample_rate = DeviceConfig.sample_rate
                    trans_fun_i.interpolate_transfer_fun()
                    mic_transfer_functions.append(trans_fun_i)
            else:
                self.ready_to_record = False
                self.logger.error(
                    'Invalid number of microphone transfer functions'
                )
                return
        else:
            mic_transfer_functions = None

        self.msrmt_ctx = MsrmtContext(
            fs=DeviceConfig.sample_rate,
            block_size=num_total_recording_samples,
            non_interactive=False,
            input_trans_fun=mic_transfer_functions,
        )
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_total_recording_samples,
            DeviceConfig.device_buffer_size
        )

        self.results = None

        self.signals = []
        self.generate_output_signals(
            msrmt_params,
            num_block_samples,
            hw_data
        )

        self.msrmt = SyncMsrmt(
            rec_data,
            hw_data,
            self.signals,
            block_duration
        )

    def record(self) -> None:
        """Starts the calibration."""

        if self.ready_to_record is False:
            return

        self.logger.info("Starting output calibration...")

        self.msrmt.run_msrmt()

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

        ch_spectra = get_channel_spectra(
            self.msrmt, self.msrmt_ctx
        )

        # plot channel spectra
        df = (
            DeviceConfig.sample_rate
            / self.msrmt_ctx.block_size
            * len(self.msrmt.hardware_data.output_channels)
        )

        # find frequency indices to sample transfer function
        mt_bin_idx = np.round((self.mt_frequencies/df)).astype(np.int_)

        max_out = []
        phase = []

        calib_results: list[npt.NDArray[np.float32]] = []
        for ch_spec in ch_spectra:
            out_db_spl = ch_spec[mt_bin_idx]

            p_out_peak = np.sqrt(2) * 20 * 10**(out_db_spl/20)
            p_out_max = p_out_peak / self.output_amplitude
            calib_results.append(p_out_max.astype(np.float32))
            max_out.append(calib_results[-1].tolist())
            phase.append(np.zeros_like(self.mt_frequencies).tolist())

        cur_time = datetime.now()
        time_stamp = cur_time.strftime("%y%m%d-%H%M%S")

        self.results = {
            "date": time_stamp,
            "output_channels": self.msrmt.hardware_data.output_channels,
            "input_channels": self.msrmt.hardware_data.input_channels,
            "frequencies": self.mt_frequencies.tolist(),
            "max_out": max_out,
            "phase": phase,
        }

    def save_recording(self) -> None:
        """Stores the measurement data in binary file."""
        if self.results is not None:
            files.save_output_calibration(self.results)

    def generate_output_signals(
        self,
        msrmt_params: CalibMsrmtParams,
        num_block_samples: int,
        hw_data: HardwareData
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

        n_total_samples = len(hw_data.output_channels) * len(mt_signal)

        counter = 0
        for i in range(hw_data.get_stream_output_channels()):
            if i in hw_data.output_channels:
                stimulus = np.zeros(n_total_samples, dtype=np.float32)
                stimulus[counter*len(mt_signal):(counter+1)*len(mt_signal)] = mt_signal
                signal = Signal(stimulus)
                self.signals.append(signal)
                counter += 1
            else:
                self.signals.append(
                    Signal(np.zeros(n_total_samples, dtype=np.float32))
                )
