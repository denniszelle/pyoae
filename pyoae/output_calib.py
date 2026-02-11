"""Classes and functions to calibrate output channels.

This module is not intended to be run directly.
"""

from collections import Counter
from datetime import datetime
from logging import Logger

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from pyoae import files
from pyoae import get_logger
from pyoae.calib_storage import (
    MicroTransferFunction,
    OutputCalibration,
    SpeakerCalibData
)
from pyoae import converter
from pyoae.device.device_config import DeviceConfig
from pyoae.msrmt_context import MsrmtContext
from pyoae import mt_generator
from pyoae.protocols import CalibMsrmtParams
from pyoae.signals import PeriodicSignal
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
        sharex='all',
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
    mt_definition: mt_generator.MultiToneDefinition
) -> list[mt_generator.MultiToneResult]:
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
        msrmt_ctx.block_size/len(sync_msrmt.hardware_data.output_channels)
    )
    block_size = msrmt_ctx.block_size

    for i, _ in enumerate(sync_msrmt.hardware_data.output_channels):

        input_channel = sync_msrmt.hardware_data.input_channels[i]
        recorded_signal = sync_msrmt.get_recorded_signal(
            input_channel
        )

        # Obtain an integer number of recorded blocks
        total_blocks = int(len(recorded_signal)/block_size)
        block_data = recorded_signal[:total_blocks*block_size]
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

        if msrmt_ctx.input_trans_fun is None:
            input_tf = None
        else:
            input_tf = msrmt_ctx.input_trans_fun[input_channel_idx]

        analyzer = mt_generator.MultiToneAnalyzer(
            mt_definition,
            block_avg[i*channel_segment_size:(i+1)*channel_segment_size],
            DeviceConfig.sample_rate,
        )

        results.append(
            analyzer.compute_result(input_tf)
        )

    return results


def get_log_frequency_ticks(
    f_min,
    f_max,
    bases=(1,3,5)
):
    """Return frequency ticks for x-axis plotting with log x-scale."""
    decade_min = int(np.floor(np.log10(f_min)))
    decade_max = int(np.ceil(np.log10(f_max)))

    decades = 10 ** np.arange(decade_min, decade_max + 1)
    ticks = np.array([b * d for d in decades for b in bases])

    return ticks[(ticks >= f_min) & (ticks <= f_max)]


def plot_offline(
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: MsrmtContext,
    mt_results: list[mt_generator.MultiToneResult]
) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.
    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return

    has_input_calib =  msrmt_ctx.input_trans_fun is not None
    f_min = np.floor((mt_results[0].frequencies.min() - 20) / 20) * 20
    f_max = np.ceil((mt_results[0].frequencies.max() + 500)/ 1000) * 1000
    f_min = max(20, f_min)
    axes = setup_offline_plot(
        (f_min, f_max),
        sync_msrmt.hardware_data.output_channels,
        sync_msrmt.hardware_data.input_channels,
        has_input_calib
    )

    padding = 15  # dB of padding on top and bottom
    phase_padding = 3 # rad of padding on top and bottom
    line_vecs = ['b-', 'r-', 'g-']
    ax_cmp_min = 80
    ax_cmp_max = 120

    input_channels = sync_msrmt.hardware_data.input_channels
    counter = Counter(input_channels)
    sorted_input_channels = list(counter.keys())

    # Get boundaries
    raw_y_min = np.inf
    raw_y_max = -np.inf
    phase_min = np.inf
    phase_max = -np.inf
    amp_min = np.inf
    amp_max = -np.inf
    for result_i in mt_results:
        raw_y_min = min(raw_y_min, np.abs(result_i.spectra).min())
        raw_y_max = max(raw_y_max, np.abs(result_i.spectra).max())
        phase_min = min(phase_min, result_i.phase.min())
        phase_max = max(phase_max, result_i.phase.max())
        amp_min = min(amp_min, result_i.amplitude.min())
        amp_max = max(amp_max, result_i.amplitude.max())
    amp_min = converter.rms_mupa_to_db_spl(amp_min) - padding
    amp_max = converter.rms_mupa_to_db_spl(amp_max) + padding
    raw_y_min = min(converter.rms_mupa_to_db_spl(raw_y_min), ax_cmp_min)
    raw_y_max = max(converter.rms_mupa_to_db_spl(raw_y_max), ax_cmp_max)
    phase_min -= phase_padding
    phase_max += phase_padding

    mt_frequencies = mt_results[0].frequencies

    for i, ax_i in enumerate(axes):
        # i is the row in the plot
        for j, ax_ij in enumerate(ax_i):
            # j is the column in the plot
            if i < len(axes) - 2:

                output_idc = np.where(
                    np.asarray(input_channels) == sorted_input_channels[j]
                )[0]
                if len(output_idc) > i:
                    output_idx = output_idc[i]
                else:
                    continue
                # Plot measurement of channel ij
                out_db_spl = converter.rms_mupa_to_db_spl(mt_results[output_idx].raw_amplitude)
                phases = mt_results[output_idx].phase

                # p_out_peak = mt_results[output_idx].raw_amplitude*np.sqrt(2)
                p_out_max = mt_results[output_idx].amplitude*np.sqrt(2)
                out_max_db_spl = converter.peak_mupa_to_db_spl(p_out_max)

                # Plot raw spectrum
                for spec_k in mt_results[output_idx].spectra:
                    ax_ij.plot(
                        mt_results[output_idx].freq_spectra,
                        converter.rms_mupa_to_db_spl(abs(spec_k)),
                        linewidth=0.5,
                        color='k'
                    )

                # Add markers to raw spectrum
                ax_ij.plot(mt_frequencies, out_db_spl, 'ro', markersize=2)

                # Add markers to raw spectrum plot with amplitude correction
                ax_ij.plot(
                    mt_frequencies,
                    out_max_db_spl,
                    'ko',
                    markersize=3,
                )

                # Add corrected values to amplitude and phase plot
                if len(line_vecs) > i:
                    axes[-2][j].plot(
                        mt_frequencies, out_max_db_spl, line_vecs[i]
                    )
                    axes[-1][j].plot(mt_frequencies, phases, line_vecs[i])
                    ax_ij.plot(mt_frequencies, out_max_db_spl, line_vecs[i])
                else:
                    axes[-2][j].plot(mt_frequencies, out_max_db_spl)
                    axes[-1][j].plot(mt_frequencies, phases)
                    ax_ij.plot(mt_frequencies, out_max_db_spl)

                ax_ij.set_ylim(raw_y_min, raw_y_max + padding)
                axes[-2][j].grid(True, which='both')
                axes[-1][j].grid(True, which='both')
                ticks = get_log_frequency_ticks(
                    min(mt_frequencies), max(mt_frequencies)
                )
                axes[-1][j].set_xticks(ticks)
                axes[-1][j].set_xticklabels([str(int(t)) for t in ticks])
                axes[-2][j].set_ylim(amp_min, amp_max)
                axes[-1][j].set_ylim(phase_min, phase_max)

    plt.tight_layout()
    plt.show()


def plot_result_file(results: OutputCalibration) -> None:
    """Plots output calibration from result file."""

    counter = Counter(results.input_channels)
    cols = len(counter)

    # Type ignore to set known dimensions of 2
    fig, axes = plt.subplots(
        2,
        cols,
        figsize=(12, 6),
        sharex='all',
        squeeze=False
    ) # type: ignore
    axes: list[list[Axes]]

    sorted_input_channels = list(counter.keys())

    line_styles = ['b.-', 'rx-', 'gd-']

    f_min = np.floor((results.raw_freqs.min() - 20) / 20) * 20
    f_max = np.ceil((results.raw_freqs.max() + 500)/ 1000) * 1000
    f_min = max(20, f_min)

    y_max = np.ceil(np.max(
        converter.peak_mupa_to_db_spl(results.raw_amps)
    )) + 10
    if y_max <= 0:
        y_min = y_max-100
    else:
        y_min = 0
    phase_max = np.ceil(np.max(results.raw_phases))+2
    phase_min = np.floor(np.min(results.raw_phases))-2

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
        ax_i_phase.set_title(
            f'Speaker Phase - Mic Channel {input_channel_i}'
        )
        ax_i_phase.set_xlabel('Frequency (Hz)')
        ax_i_phase.set_ylabel('Phase (rad)')


        output_idc = np.where(
            np.asarray(results.input_channels) == input_channel_i
        )[0]
        for j, output_idx_j in enumerate(output_idc):
            output_channel_j = results.output_channels[output_idx_j]

            p_out_max = results.raw_amps[output_idx_j,:]
            out_max_db_spl = converter.peak_mupa_to_db_spl(p_out_max)
            if i < len(line_styles):
                ax_i_amp.plot(
                    results.raw_freqs,
                    out_max_db_spl,
                    line_styles[j],
                    label=f'Channel {output_channel_j} Maximum Output Level'
                )
                ax_i_phase.plot(
                    results.raw_freqs,
                    results.raw_phases[output_idx_j],
                    line_styles[j],
                    label=f'Channel {output_channel_j} Speaker Phase'
                )
            else:
                ax_i_amp.plot(
                    results.raw_freqs,
                    out_max_db_spl,
                    label=f'Channel {output_channel_j}'
                )
                ax_i_phase.plot(
                    results.raw_freqs,
                    results.raw_phases[output_idx_j],
                    label=f'Channel {output_channel_j} Speaker Phase'
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

    mt_definition: mt_generator.MultiToneDefinition

    mt_results: list[mt_generator.MultiToneResult]

    signals: list[PeriodicSignal]
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
            msrmt_params['block_duration']
            * len(output_channels)
            * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            num_block_samples
            * msrmt_params['num_averaging_blocks']
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
            block_size=num_block_samples,
            non_interactive=False,
            input_trans_fun=mic_transfer_functions,
        )
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_block_samples,
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
            plot_offline(
                self.msrmt,
                self.msrmt_ctx,
                self.mt_results
            )

    def compute_calib_results(self) -> None:
        """Computes the output-channel transfer functions."""
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
                (result_i.amplitude*np.sqrt(2)).astype(float).tolist()
            )
            phase.append(result_i.phase.astype(float).tolist())

        cur_time = datetime.now()
        time_stamp = cur_time.strftime('%y%m%d-%H%M%S')

        self.results = {
            'date': time_stamp,
            'output_channels': self.msrmt.hardware_data.output_channels,
            'input_channels': self.msrmt.hardware_data.input_channels,
            'frequencies': frequencies,
            'max_out': max_out,
            'phase': phase
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
        mt_samples = int(np.round(
            num_block_samples
            / len(hw_data.output_channels)
        ))
        df = DeviceConfig.sample_rate/mt_samples*msrmt_params['num_clusters']
        mt_frequencies = mt_generator.compute_mt_frequencies(
            msrmt_params['f_start'],
            msrmt_params['f_stop'],
            msrmt_params['lines_per_octave'],
            df
        )
        # Remove redundant frequencies
        mt_frequencies = np.unique(mt_frequencies)
        num_mt_frequencies = len(mt_frequencies)
        mt_phases = mt_generator.compute_mt_phases(num_mt_frequencies)
        amplitudes = (
            np.ones_like(mt_frequencies)
            * msrmt_params['amplitude_per_line']
        ).astype(np.float32)
        cluster_idc = np.arange(num_mt_frequencies)
        cluster_idc = cluster_idc % msrmt_params['num_clusters']


        self.mt_definition = mt_generator.MultiToneDefinition(
            mt_frequencies,
            mt_phases,
            amplitudes,
            cluster_idc.astype(np.int32)
        )



        # mt_signal = mt_generator.generate_mt_signal(
        #     mt_samples,
        #     DeviceConfig.sample_rate,
        #     self.mt_frequencies,
        #     self.mt_phases
        # )

        mt_signal = self.mt_definition.generate_mt_signal(
            mt_samples,
            DeviceConfig.sample_rate,
            DeviceConfig.ramp_duration
        )

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

        n_total_samples = num_block_samples * msrmt_params['num_averaging_blocks']

        counter = 0
        for i in range(hw_data.get_stream_output_channels()):
            if i in hw_data.output_channels:
                stimulus = np.zeros(num_block_samples, dtype=np.float32)
                stimulus[
                    counter*len(mt_signal):(counter+1)*len(mt_signal)
                ] = mt_signal
                signal = PeriodicSignal(stimulus, n_total_samples)
                self.signals.append(signal)
                counter += 1
            else:
                self.signals.append(
                    PeriodicSignal()
                )
