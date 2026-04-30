"""Classes and functions to calibrate output channels.

This module is not intended to be run directly.
"""

from collections import Counter
from datetime import datetime
from logging import Logger
from typing import TypedDict

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt

from pyoae import files
from pyoae import get_logger
from pyoae.calib_storage import SpeakerCalibData

from pyoae.calib_transfer import (
    EarSimTransferFunction,
    MicroTransferFunction,
    OutputCalibration
)
from pyoae import converter
from pyoae.device.device_config import DeviceConfig
from pyoae.msrmt_context import MsrmtContext
from pyoae.mt_generator import (
    MultiToneAnalyzer,
    MultiToneDefinition,
    MultiToneResult,
)
from pyoae.protocols import CalibMsrmtParams, CalibMsrmtDef
from pyoae.signals import PeriodicSignal
from pyoae.sync import (
    get_input_channels,
    HardwareData,
    RecordingData,
    SyncMsrmt,
    MsrmtState,
)

logger = get_logger(__name__)


class PlotConfig(TypedDict):
    """Configuration of plotting"""

    axes: list[list[Axes]]
    """2D list of axes objects for plot grid"""

    frequencies: npt.NDArray[np.floating]
    """Frequencies of axes"""

    input_channels: list[int]
    """Input channels for speakers to calibrate"""

    sorted_input_channels: list[int]
    """Sorted input channels for speakers to calibrate"""

    line_vecs: list[str]
    """Line plot options for different speaker channels on the same input"""

    padding: float
    """Distances between data- and plot boundaries for amplitude plots"""

    phase_padding: float
    """Distances between data- and plot boundaries for phase plots"""

    ax_cmp_min: float
    """Maximum value for lower boundaries of raw plots"""

    ax_cmp_max: float
    """Minimum value for upper boundaries of raw plots"""


class PlotBounds(TypedDict):
    """Boundary information of plots"""

    raw_y_min: float
    """Minimum amplitude of raw data plots"""

    raw_y_max: float
    """Maximum amplitude of raw data plots"""

    phase_min: float
    """Minimum phase of raw data plots"""

    phase_max: float
    """Maximum phase of raw data plots"""

    amp_min: float
    """Minimum amplitude of corrected data"""

    amp_max: float
    """Maximum amplitude of corrected data"""


def setup_offline_plot(
    frequency_range: tuple[float, float],
    output_channels: list[int],
    input_channels: list[int],
    is_calib_available: bool = False,
) -> list[list[Axes]]:
    """Sets up the plots.

    Args:
        recording_duration: Total duration of the recording in seconds
        fs: Sampling frequency in Hz
        block_size: Size of each measurement block that is repeated
          periodically
        is_calib_available: Boolean whether a calibration is available to
          display sound pressure or only show raw measurement data

    Returns:
        list[axes]

        - **fig**: Object containing the plots
        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    counter = Counter(input_channels)
    rows = max(counter.values()) + 2
    cols = len(counter)

    # Type ignore to set known dimensions of 2
    _, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 8),
        sharex='col',
        squeeze=False
    ) # type: ignore
    axes: list[list[Axes]]

    sorted_input_channels = list(counter.keys())

    for i, ax_i in enumerate(axes):
        for j, ax_ij in enumerate(ax_i):
            if i == len(axes) - 2:
                axes[-2][j].set_title(
                    'Channel Comparison for Input '
                    f'Channel {sorted_input_channels[j]}'
                )
            elif i == len(axes) - 1:
                axes[-1][j].set_title(
                    'Channel Phase Comparison for Input '
                    f'Channel {sorted_input_channels[j]}'
                )
                axes[-1][j].set_xlabel('Frequency (Hz)')
            else:
                output_idc = np.where(
                    np.asarray(input_channels) == sorted_input_channels[j]
                )[0]
                if len(output_idc) > i:
                    output_channel_ij = output_channels[output_idc[i]]
                    ax_ij.set_xlim(frequency_range[0], frequency_range[1])
                    ax_ij.set_ylim(-50, 100)
                    ax_ij.set_xscale('log')
                    ax_ij.set_title(
                        f'Spectrum of Output Channel {output_channel_ij}'
                    )
                    if is_calib_available:
                        ax_ij.set_ylabel('Level (dB SPL)')
                    else:
                        ax_ij.set_ylabel('Level (dBFS)')
    return axes


def get_mt_results(
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    mt_definition: MultiToneDefinition,
) -> list[MultiToneResult]:
    """Processes data and returns spectra per output channels.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.

    Returns:
        list[spectrum_ch01, spectrum_ch02, ...]
    """

    results = []

    channel_segment_size = int(
        msrmt_ctx.block_size / len(sync_msrmt.hardware_data.output_channels)
    )
    block_size = msrmt_ctx.block_size

    for i, _ in enumerate(sync_msrmt.hardware_data.output_channels):

        input_channel = sync_msrmt.hardware_data.input_channels[i]
        recorded_signal = sync_msrmt.get_recorded_signal(input_channel)

        # Obtain an integer number of recorded blocks
        total_blocks = int(len(recorded_signal) / block_size)
        block_data = recorded_signal[: total_blocks * block_size]
        blocks = block_data.reshape(-1, block_size)

        if input_channel not in sync_msrmt.hardware_data.input_channels:
            logger.error('Invalid input channel')
            results.append(np.zeros(0, dtype=np.float32))
            continue

        input_channel_idx = (
            sync_msrmt.hardware_data.get_unique_input_channels().index(
                input_channel
            )
        )

        block_avg = np.mean(blocks, axis=0)

        if msrmt_ctx.mic_trans_fun is None:
            mic_tf = None
        else:
            mic_tf = msrmt_ctx.mic_trans_fun[input_channel_idx]
        if msrmt_ctx.ear_sim_trans_fun is None:
            ear_sim_tf = None
        else:
            ear_sim_tf = msrmt_ctx.ear_sim_trans_fun[input_channel_idx]

        analyzer = MultiToneAnalyzer(
            mt_definition,
            block_avg[
                i * channel_segment_size : (i + 1) * channel_segment_size
            ],
            DeviceConfig.sample_rate,
        )

        results.append(analyzer.compute_result(mic_tf, ear_sim_tf))

    return results


def get_log_frequency_ticks(f_min, f_max, bases=(1, 3, 5)):
    """Generate frequency tick positions for a logarithmic x-axis.

    Creates a set of tick positions suitable for plotting frequencies
    on a logarithmic scale, optionally including additional bases within
    each decade.

    Args:
        f_min (float): Minimum frequency for the axis (Hz).
        f_max (float): Maximum frequency for the axis (Hz).
        bases (tuple of int, optional): Multipliers within each decade to
            include as ticks. Defaults to (1, 3, 5).

    Returns:
        np.ndarray: Array of frequency tick positions within the range [f_min, f_max].
    """
    decade_min = int(np.floor(np.log10(f_min)))
    decade_max = int(np.ceil(np.log10(f_max)))

    decades = 10 ** np.arange(decade_min, decade_max + 1)
    ticks = np.array([b * d for d in decades for b in bases])

    return ticks[(ticks >= f_min) & (ticks <= f_max)]


def _prepare_plot_config(
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    mt_results: list[MultiToneResult],
) -> PlotConfig:
    """Create plot config containing meta information of the recorded data."""

    has_input_calib = msrmt_ctx.mic_trans_fun is not None

    f_min = np.floor((mt_results[0].frequencies.min() - 20) / 20) * 20
    f_max = np.ceil((mt_results[0].frequencies.max() + 500) / 1000) * 1000
    f_min = max(20, f_min)

    axes = setup_offline_plot(
        (f_min, f_max),
        sync_msrmt.hardware_data.output_channels,
        sync_msrmt.hardware_data.input_channels,
        has_input_calib,
    )

    input_channels = sync_msrmt.hardware_data.input_channels
    sorted_input_channels = list(Counter(input_channels).keys())

    return {
        'axes': axes,
        'frequencies': mt_results[0].frequencies,
        'input_channels': input_channels,
        'sorted_input_channels': sorted_input_channels,
        'line_vecs': ['b-', 'r-', 'g-'],
        'padding': 15.0,
        'phase_padding': 3.0,
        'ax_cmp_min': 80.0,
        'ax_cmp_max': 120.0,
    }


def _compute_bounds(
    mt_results: list[MultiToneResult], config: PlotConfig
) -> PlotBounds:
    """Compute x- and y-value boundaries for plots"""

    raw_y_min, raw_y_max = np.inf, -np.inf
    phase_min, phase_max = np.inf, -np.inf
    amp_min, amp_max = np.inf, -np.inf

    for r in mt_results:
        raw_y_min = min(raw_y_min, np.abs(r.spectra).min())
        raw_y_max = max(raw_y_max, np.abs(r.spectra).max())
        phase_min = min(phase_min, r.phase.min())
        phase_max = max(phase_max, r.phase.max())
        amp_min = min(amp_min, r.amplitude.min())
        amp_max = max(amp_max, r.amplitude.max())

    padding = config['padding']

    amp_min = converter.rms_mupa_to_db_spl(amp_min) - padding
    amp_max = converter.rms_mupa_to_db_spl(amp_max) + padding

    raw_y_min = min(
        converter.rms_mupa_to_db_spl(raw_y_min), config['ax_cmp_min']
    )
    raw_y_max = max(
        converter.rms_mupa_to_db_spl(raw_y_max), config['ax_cmp_max']
    )

    phase_min -= config['phase_padding']
    phase_max += config['phase_padding']

    return {
        'raw_y_min': raw_y_min,
        'raw_y_max': raw_y_max,
        'phase_min': phase_min,
        'phase_max': phase_max,
        'amp_min': amp_min,
        'amp_max': amp_max,
    }


def _get_output_index(config, i, j):
    input_channels = config['input_channels']
    sorted_input_channels = config['sorted_input_channels']

    output_idc = np.where(
        np.asarray(input_channels) == sorted_input_channels[j]
    )[0]

    if len(output_idc) > i:
        return output_idc[i]

    return None


def _apply_axis_formatting(
    ax: Axes,
    axes: list[list[Axes]],
    j: int,
    config: PlotConfig,
    bounds: PlotBounds,
):
    """Apply axis formatting for plots."""
    ax.set_ylim(bounds['raw_y_min'], bounds['raw_y_max'] + config['padding'])

    axes[-2][j].grid(True, which='both')
    axes[-1][j].grid(True, which='both')

    ticks = get_log_frequency_ticks(
        min(config['frequencies']), max(config['frequencies'])
    )

    axes[-1][j].set_xticks(ticks)
    axes[-1][j].set_xticklabels([str(int(t)) for t in ticks])

    axes[-2][j].set_ylim(bounds['amp_min'], bounds['amp_max'])
    axes[-1][j].set_ylim(bounds['phase_min'], bounds['phase_max'])


def _plot_single_channel(
    ax: Axes,
    axes: list[list[Axes]],
    i: int,
    j: int,
    result: MultiToneResult,
    config: PlotConfig,
    bounds: PlotBounds,
    ear_sim_tf: EarSimTransferFunction | None,
):
    """Add plots for a single channel of an output calibration."""

    freqs = config['frequencies']
    line_vecs = config['line_vecs']

    out_db_spl = converter.rms_mupa_to_db_spl(result.raw_amplitude)
    phases = result.phase

    p_out_max = result.amplitude * np.sqrt(2)
    out_max_db_spl = converter.peak_mupa_to_db_spl(p_out_max)

    if ear_sim_tf is not None:
        cplx_ear_sim_vals = ear_sim_tf.get_interp_transfer_function(freqs)
        out_max_db_spl_uncorrected = (
            out_max_db_spl - converter.lin_to_db(np.abs(cplx_ear_sim_vals))
        )
        phases_uncorreceted = (
            phases - np.angle(cplx_ear_sim_vals)
        )
    else:
        out_max_db_spl_uncorrected = None
        phases_uncorreceted = None

    # Add raw data to own raw-data plot
    for spec in result.spectra:
        ax.plot(
            result.freq_spectra,
            converter.rms_mupa_to_db_spl(abs(spec)),
            linewidth=0.5,
            color='k'
        )

    # Add markers to raw data plot
    ax.plot(freqs, out_db_spl, 'ro', markersize=2)
    ax.plot(freqs, out_max_db_spl, 'ko', markersize=3)

    # Define style of plots in comparison plots
    style = line_vecs[i] if i < len(line_vecs) else None

    # Add results to amplitude comparison(-2) and phase comparison(-1) plots
    if style:
        axes[-2][j].plot(freqs, out_max_db_spl, style)
        if out_max_db_spl_uncorrected is not None:
            axes[-2][j].plot(
                freqs,
                out_max_db_spl_uncorrected,
                style,
                alpha=0.4,
                linewidth=0.9
            )

        axes[-1][j].plot(freqs, phases, style)
        if phases_uncorreceted is not None:
            axes[-1][j].plot(
                freqs, phases_uncorreceted, style, alpha=0.4, linewidth=0.9
            )

        ax.plot(freqs, out_max_db_spl, style)
    else:
        axes[-2][j].plot(freqs, out_max_db_spl)
        if out_max_db_spl_uncorrected is not None:
            axes[-2][j].plot(freqs, out_max_db_spl_uncorrected, alpha=0.5)
        axes[-1][j].plot(freqs, phases)
        if phases_uncorreceted is not None:
            axes[-1][j].plot(freqs, phases_uncorreceted, alpha=0.5)
        ax.plot(freqs, out_max_db_spl)

    # Format axes
    _apply_axis_formatting(ax, axes, j, config, bounds)


def plot_offline(
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    mt_results: list[MultiToneResult],
) -> None:
    """Create offline plot after a output calibration measurement.

    Measurements for each input that are assigned to an output channel that
    is calibrated, one row is created. The first plots display raw data for
    each speaker recording. The second last plot plots all (corrected)
    amplitudes in a single plot, the last one all (corrected phases) in one.
    """

    if sync_msrmt.state != MsrmtState.FINISHED:
        return

    config = _prepare_plot_config(sync_msrmt, msrmt_ctx, mt_results)
    bounds = _compute_bounds(mt_results, config)

    axes = config["axes"]

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            if i >= len(axes) - 2:
                continue

            output_idx = _get_output_index(config, i, j)
            if output_idx is None:
                continue

            if msrmt_ctx.ear_sim_trans_fun is None:
                ear_sim_tf = None
            else:
                ear_sim_tf = msrmt_ctx.ear_sim_trans_fun[j]

            _plot_single_channel(
                ax,
                axes,
                i,
                j,
                mt_results[output_idx],
                config,
                bounds,
                ear_sim_tf
            )

    plt.tight_layout()
    plt.show()


def plot_result_file(
    results: OutputCalibration,
    ear_sim_tfs: list[EarSimTransferFunction] | None
) -> None:
    """Plots output calibration from result file."""

    counter = Counter(results.input_channels)
    cols = len(counter)

    # Type ignore to set known dimensions of 2
    fig, axes = plt.subplots(
        2, cols, figsize=(12, 6), sharex='all', squeeze=False
    )  # type: ignore
    axes: list[list[Axes]]

    sorted_input_channels = list(counter.keys())

    line_styles = ['b.-', 'rx-', 'gd-']

    f_min = np.floor((results.raw_freqs.min() - 20) / 20) * 20
    f_max = np.ceil((results.raw_freqs.max() + 500) / 1000) * 1000
    f_min = max(20, f_min)

    y_max = (
        np.ceil(np.max(converter.peak_mupa_to_db_spl(results.raw_amps))) + 10
    )
    if y_max <= 0:
        y_min = y_max - 100
    else:
        y_min = 0
    phase_max = np.ceil(np.max(results.raw_phases)) + 2
    phase_min = np.floor(np.min(results.raw_phases)) - 2

    for i, input_channel_i in enumerate(sorted_input_channels):

        ax_i_amp = axes[0][i]
        ax_i_amp.set_xlim(f_min, f_max)
        ax_i_amp.set_xscale('log')
        ax_i_amp.set_title(
            f'Maximum Output Level - Mic Channel {input_channel_i}'
        )
        ax_i_amp.set_ylabel('Level (dB SPL)')
        ax_i_phase = axes[1][i]
        ax_i_amp.set_xlim(f_min, f_max)
        ax_i_phase.set_xscale('log')
        ax_i_phase.set_title(f'Speaker Phase - Mic Channel {input_channel_i}')
        ax_i_phase.set_xlabel('Frequency (Hz)')
        ax_i_phase.set_ylabel('Phase (rad)')

        output_idc = np.where(
            np.asarray(results.input_channels) == input_channel_i
        )[0]
        for j, output_idx_j in enumerate(output_idc):
            output_channel_j = results.output_channels[output_idx_j]

            p_out_max = results.raw_amps[output_idx_j, :]
            out_max_db_spl = converter.peak_mupa_to_db_spl(p_out_max)

            if ear_sim_tfs is not None:
                cplx_ear_sim_vals = ear_sim_tfs[i].get_interp_transfer_function(
                    results.raw_freqs
                )
                out_max_db_spl_uncorrected = (
                    out_max_db_spl - converter.lin_to_db(np.abs(cplx_ear_sim_vals))
                )
                phases_uncorreceted = (
                    results.raw_phases[output_idx_j] - np.angle(cplx_ear_sim_vals)
                )
            else:
                out_max_db_spl_uncorrected = None
                phases_uncorreceted = None

            if i < len(line_styles):
                ax_i_amp.plot(
                    results.raw_freqs,
                    out_max_db_spl,
                    line_styles[j],
                    label=f'Channel {output_channel_j} Maximum Output Level',
                )
                if out_max_db_spl_uncorrected is not None:
                    ax_i_amp.plot(
                        results.raw_freqs,
                        out_max_db_spl_uncorrected,
                        line_styles[j],
                        alpha=0.4,
                        linewidth=0.9
                    )
                ax_i_phase.plot(
                    results.raw_freqs,
                    results.raw_phases[output_idx_j],
                    line_styles[j],
                    label=f'Channel {output_channel_j} Speaker Phase',
                )
                if phases_uncorreceted is not None:
                    ax_i_phase.plot(
                        results.raw_freqs,
                        phases_uncorreceted,
                        line_styles[j]
                    )
            else:
                ax_i_amp.plot(
                    results.raw_freqs,
                    out_max_db_spl,
                    label=f'Channel {output_channel_j}',
                )
                if out_max_db_spl_uncorrected is not None:
                    ax_i_amp.plot(
                        results.raw_freqs,
                        out_max_db_spl_uncorrected,
                        alpha=0.4,
                        linewidth=0.9
                    )
                ax_i_phase.plot(
                    results.raw_freqs,
                    results.raw_phases[output_idx_j],
                    label=f'Channel {output_channel_j} Speaker Phase',
                )
                if phases_uncorreceted is not None:
                    ax_i_phase.plot(
                        results.raw_freqs,
                        phases_uncorreceted,
                    )


        ax_i_amp.set_ylim(y_min, y_max)
        ax_i_phase.set_ylim(phase_min, phase_max)

        ticks = get_log_frequency_ticks(
            min(results.raw_freqs), max(results.raw_freqs)
        )
        ax_i_phase.set_xticks(ticks)
        ax_i_phase.set_xticklabels([str(int(t)) for t in ticks])

        ax_i_amp.legend()
        ax_i_amp.grid(True, which='both')
        ax_i_phase.grid(True, which='both')
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f'Calibration {results.date}')
    plt.tight_layout()
    plt.show()


class OutputCalibRecorder:
    """Class to manage a DPOAE recording."""

    mt_definition: MultiToneDefinition
    """Definition of multitone signals"""

    mt_results: list[MultiToneResult]
    """Results of multitone measurements"""

    signals: list[PeriodicSignal]
    """List of output signals for each channel."""

    msrmt_ctx: MsrmtContext
    """Instance to perform a synchronized OAE measurement."""

    msrmt: SyncMsrmt | None
    """Instance to perform a synchronized measurement."""

    results: SpeakerCalibData | None
    """Calibration results for output channels."""

    logger: Logger
    """Class logger for debug, info, warning, and error messages."""

    def __init__(
        self,
        msrmt_params: CalibMsrmtParams | CalibMsrmtDef,
        output_channels: list[int],
        mic_trans_fun: list[MicroTransferFunction] | None = None,
        ear_sim_trans_fun: list[EarSimTransferFunction] | None = None,
        log: Logger | None = None,
    ) -> None:
        """Creates a simple multi-tone output calibrator."""

        self.logger = log or get_logger()
        num_block_samples = int(
            msrmt_params['block_duration']
            * len(output_channels)
            * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            num_block_samples * msrmt_params['num_averaging_blocks']
        )
        block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = (
            num_total_recording_samples / DeviceConfig.sample_rate
        )

        # Set to false if major problem occured during calibration
        self.results = None

        if block_duration == msrmt_params['block_duration'] * len(
            output_channels
        ):
            self.logger.info(
                'Block duration adjusted to %.2f ms.', block_duration * 1E3
            )
        else:
            self.logger.warning(
                'Block duration set to %.2f ms.', block_duration * 1E3
            )

        # Setup hardware data
        active_in_channels = list(
            {
                b for a, b in DeviceConfig.output_input_mapping
                if a in output_channels
            }
        )

        n_in_channels = (
            max(*active_in_channels, DeviceConfig.sync_channels[1]) + 1
        )
        n_out_channels = max(output_channels) + 1
        hw_data = HardwareData(
            n_in_channels,
            n_out_channels,
            DeviceConfig.input_device,
            DeviceConfig.output_device,
            output_channels,
            get_input_channels(output_channels),
        )

        if mic_trans_fun:
            mic_transfer_functions = []
            if len(mic_trans_fun) == len(active_in_channels):
                for trans_fun_i in mic_trans_fun:
                    mic_transfer_functions.append(trans_fun_i)
            else:
                self.msrmt = None
                self.logger.error(
                    'Invalid number of microphone transfer functions'
                )
                return
        else:
            mic_transfer_functions = None

        if ear_sim_trans_fun:
            ear_sim_transfer_functions = []
            if len(ear_sim_trans_fun) == len(active_in_channels):
                for trans_fun_i in ear_sim_trans_fun:
                    ear_sim_transfer_functions.append(trans_fun_i)
            else:
                self.msrmt = None
                self.logger.error(
                    'Invalid number of ear simulator transfer functions'
                )
                return
        else:
            ear_sim_transfer_functions = None

        self.msrmt_ctx = MsrmtContext(
            fs=DeviceConfig.sample_rate,
            block_size=num_block_samples,
            non_interactive=False,
            mic_trans_fun=mic_transfer_functions,
            ear_sim_trans_fun=ear_sim_transfer_functions
        )
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_block_samples,
            DeviceConfig.device_buffer_size,
        )

        self.results = None

        self.signals = []

        is_generated = self.generate_output_signals(
            msrmt_params, num_block_samples, hw_data
        )
        if not is_generated:
            self.msrmt = None
            return

        self.msrmt = SyncMsrmt(rec_data, hw_data, self.signals, block_duration)

    def record(self) -> None:
        """Starts the calibration."""

        if self.msrmt is None:
            return

        self.logger.info('Starting output calibration...')

        self.msrmt.run_msrmt()

        # # # Compute calibration results
        self.compute_calib_results()

        if self.results is None:
            return

        if not self.msrmt_ctx.non_interactive:
            # Plot all data and final result after user has
            # closed the live-measurement window.
            self.logger.info(
                'Showing offline results. Please close window to continue.'
            )
            plot_offline(self.msrmt, self.msrmt_ctx, self.mt_results)

    def compute_calib_results(self) -> None:
        """Computes the output-channel transfer functions."""
        if self.msrmt is None:
            return

        if self.msrmt.state != MsrmtState.FINISHED:
            return

        self.mt_results = get_mt_results(
            self.msrmt, self.msrmt_ctx, self.mt_definition
        )

        max_out = []
        phase = []

        frequencies = self.mt_results[0].frequencies.astype(float).tolist()

        for result_i in self.mt_results:
            max_out.append(
                (result_i.amplitude * np.sqrt(2)).astype(float).tolist()
            )
            phase.append(result_i.phase.astype(float).tolist())

        cur_time = datetime.now()
        time_stamp = cur_time.strftime('%y%m%d-%H%M%S')

        if self.msrmt_ctx.ear_sim_trans_fun is None:
            has_ear_sim_tf_included = False
            ear_sim_frequencies = None
            ear_sim_amplitudes = None
            ear_sim_phases = None
        else:
            has_ear_sim_tf_included = True
            ear_sim_frequencies = []
            ear_sim_amplitudes = []
            ear_sim_phases = []
            for ear_sim_trans_fun_i in self.msrmt_ctx.ear_sim_trans_fun:
                ear_sim_frequencies.append(
                    ear_sim_trans_fun_i.raw_freqs.tolist()
                )
                ear_sim_amplitudes.append(
                    ear_sim_trans_fun_i.raw_amps.tolist()
                )
                ear_sim_phases.append(
                    ear_sim_trans_fun_i.raw_phases.tolist()
                )


        self.results = {
            'date': time_stamp,
            'output_channels': self.msrmt.hardware_data.output_channels,
            'input_channels': self.msrmt.hardware_data.input_channels,
            'frequencies': frequencies,
            'max_out': max_out,
            'phase': phase,
            'has_ear_sim_tf_included': has_ear_sim_tf_included,
            'ear_sim_frequencies': ear_sim_frequencies,
            'ear_sim_amplitudes': ear_sim_amplitudes,
            'ear_sim_phases': ear_sim_phases,
        }

    def save_recording(self) -> None:
        """Stores the measurement data in binary file."""
        if self.results is not None:
            files.save_output_calibration(self.results)

    def generate_output_signals(
        self,
        msrmt_params: CalibMsrmtParams | CalibMsrmtDef,
        num_block_samples: int,
        hw_data: HardwareData,
    ) -> bool:
        """Generate multi-tone output signals for playback.

        Args:
            msrmt_params (CalibMsrmtParams | CalibMsrmtDef):
                Measurement parameters defining frequencies, phases, amplitudes,
                and clustering information for multi-tone calibration.
            num_block_samples (int): Number of samples per block for each output
                channel.
            hw_data (HardwareData): Hardware information including active output
                channels.

        Returns:
            bool: True if signal generation succeeded, False if an error occurred
                (e.g., invalid protocol type).
        """
        mt_samples = int(
            np.round(num_block_samples / len(hw_data.output_channels))
        )

        self.mt_definition = MultiToneDefinition(msrmt_params)

        mt_signal = self.mt_definition.generate_mt_signal(
            mt_samples, DeviceConfig.sample_rate, DeviceConfig.ramp_duration
        )

        max_amplitude = np.max(mt_signal)
        if max_amplitude > DeviceConfig.max_digital_output:
            self.logger.warning(
                'Maximum output %.2f limited to maximum %.2f re FS.',
                max_amplitude,
                DeviceConfig.max_digital_output,
            )
            self.logger.warning('Output calibration results might be invalid.')

        n_total_samples = (
            num_block_samples * msrmt_params['num_averaging_blocks']
        )

        counter = 0
        for i in range(hw_data.get_stream_output_channels()):
            if i in hw_data.output_channels:
                stimulus = np.zeros(num_block_samples, dtype=np.float32)
                stimulus[
                    counter * len(mt_signal) : (counter + 1) * len(mt_signal)
                ] = mt_signal
                signal = PeriodicSignal(stimulus, n_total_samples)
                self.signals.append(signal)
                counter += 1
            else:
                self.signals.append(PeriodicSignal())
        return True
