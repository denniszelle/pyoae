"""Module with classes and functions to process and visualize recordings."""

from logging import Logger
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt

from pyoae import files
from pyoae import generator
from pyoae import get_logger
from pyoae.calib import MicroTransferFunction
from pyoae.dsp import filters
from pyoae.dsp import processing
from pyoae.dsp.containers import (
    DpoaeMsrmtData,
    PulseDpoaeRecording
)
from pyoae.dsp.opt_avg import OptAverage


ROI_BOUNDARIES = [1750.0, 3500.0]
"""f2 frequency boundaries for region of interest lengths."""

ROI_LENGTHS = [50.0, 40.0, 30.0]
"""Durations of regions of interest in milliseconds."""


def _msrmt_to_pulse_recording(
    msrmt_data: DpoaeMsrmtData
) -> PulseDpoaeRecording:
    """Creates a pulsed DPOAE recording dictionary from measurement data."""
    return {
        'recording': msrmt_data,
        'average': None,
        'signal': None
    }


def _get_roi_length(f2: float) -> float:
    """Retrieves the length of analysis segment for a pulsed DPOAE.

    The DPOAE latency depends on the specified f2 frequency with
    lower frequencies resulting in longer latencies (specified in ms).

    For a region of interest (ROI) centered around the pulsed DPOAE,
    different minimum lengths are required to capture the complete
    DPOAE response.

    Args:
        f2: frequency of second primary tone in Hz

    Returns:
        - length of region of interest in ms
    """
    if f2 < ROI_BOUNDARIES[0]:
        return ROI_LENGTHS[0]
    if f2 < ROI_BOUNDARIES[1]:
        return ROI_LENGTHS[1]
    return ROI_LENGTHS[2]


class PulseDpoaeResult:
    """Instance to manage a DPOAE result from pulsed recording."""

    log: Logger

    recording: DpoaeMsrmtData

    raw_averaged: npt.NDArray[np.float64]

    dpoae_signal: npt.NDArray[np.float64]

    def __init__(self, pulsed_recording: PulseDpoaeRecording) -> None:
        self.log = get_logger(__class__.__name__)
        self.recording = pulsed_recording['recording']
        if pulsed_recording['average'] is None:
            self.raw_averaged = np.empty(0, dtype=np.float64)
        else:
            self.raw_averaged = pulsed_recording['average']
        if pulsed_recording['signal'] is None:
            self.dpoae_signal = np.empty(0, dtype=np.float64)
        else:
            self.dpoae_signal = pulsed_recording['signal']

    def plot(self, block_loop: bool = True) -> None:
        """Plots the pulsed DPOAE data."""
        if self.recording is None:
            return

        samplerate = self.recording['samplerate']
        num_recording_samples = len(self.recording['recorded_signal'])
        fig, axes = plt.subplots(3, 1, figsize=(10, 6))
        axes: list[Axes]

        # plot recording overview
        t_rec = np.arange(num_recording_samples) / samplerate
        axes[0].plot(t_rec, self.recording['recorded_signal'], linewidth=0.5)
        # axes[0].plot(t_rec, self.filtered_recording, linewidth=0.5)

        t_avg = np.arange(self.recording['num_block_samples']) / samplerate * 1E3
        if self.raw_averaged.size:
            axes[1].plot(t_avg, self.raw_averaged, linewidth=0.5)
        if self.dpoae_signal.size:
            axes[2].plot(t_avg, self.dpoae_signal, linewidth=0.5)

        y_lim = np.ceil(np.max(np.abs(self.dpoae_signal))/50)*50
        y_lim = max(y_lim, 50)

        axes[0].set_xlim(0, t_rec[-1])
        axes[1].set_xlim(0, t_avg[-1])
        axes[2].set_xlim(0, t_avg[-1])
        axes[2].set_ylim(-y_lim, y_lim)
        axes[0].set_xlabel("Recording Time (s)")
        axes[1].set_ylabel('Amp. (full scale)')
        axes[2].set_ylabel('p (muPa)')
        axes[2].set_xlabel('t (ms)')

        # rec_lim = axes[0].get_ylim()
        # axes[1].set_ylim(rec_lim)
        axes[0].set_title(
            f'L1: {self.recording["level1"]} dB SPL, '
            f'L2: {self.recording["level2"]} dB SPL, '
            f'f2: {self.recording["f2"]} Hz, '
            f'f2/f1: {self.recording["f2"]/self.recording["f1"]}'
        )
        axes[1].set_title('Raw Average')
        dpoae_max = np.max(self.dpoae_signal)
        axes[2].set_title(
            f'Filtered Average - DPOAE Signal (Max: {dpoae_max:.2f})'
        )
        fig.tight_layout()
        plt.show(block=block_loop)


class PulseDpoaeProcessor(PulseDpoaeResult):
    """Instance to process a pulsed DPOAE recording."""

    averager: OptAverage

    filtered_recording: npt.NDArray[np.float64]

    mic_trans_fun: MicroTransferFunction | None

    def __init__(
        self,
        msrmt_data: DpoaeMsrmtData,
        mic: MicroTransferFunction | None = None,
        mic_path: str | Path | None = None
    ) -> None:
        """Initialize processor and load recording."""
        pulsed_recording = _msrmt_to_pulse_recording(msrmt_data)
        super().__init__(pulsed_recording)
        self.averager = OptAverage()
        self.filtered_recording = np.empty(0, dtype=np.float64)
        if mic_path:
            mic_calib_data = files.load_micro_calib(mic_path)
            if mic_calib_data is not None:
                self.mic_trans_fun = MicroTransferFunction(
                    mic_calib_data['abs_calibration'],
                    mic_calib_data['transfer_function']
                )
        else:
            self.mic_trans_fun = mic

    def process_msrmt(self) -> None:
        """Process measurement to extract DPOAE."""
        if self.recording is None:
            return

        samplerate = self.recording['samplerate']
        recorded_signal = self.recording['recorded_signal']
        recorded_signal -= np.mean(recorded_signal)  # remove DC
        # perform high-pass filtering
        self.filtered_recording = filters.high_pass_filter(
            recorded_signal,
            filters.scale_filter_order(filters.HP_ORDER, samplerate),
            samplerate
        )
        fdp = 2 * self.recording['f1'] - self.recording['f2']
        if self.mic_trans_fun is None:
            self.log.warning(
                'Microphone data missing. Falling back to unity conversion.'
            )
            s = 1
        else:
            s = self.mic_trans_fun.get_sensitivity(fdp)

        self.raw_averaged = self.process_raw_data(
            self.filtered_recording,
            self.recording['num_block_samples']
        )
        self.raw_averaged /= s  # convert to muPa

        # apply ramp at edges to avoid edge effects
        ramp_len = int(
            filters.RAMP_DURATION * 1E-3 * samplerate
        )
        ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
        ramp = ramp.astype(np.float32)
        win = np.ones_like(self.raw_averaged)
        win[:ramp_len] = ramp
        win[-ramp_len:] = ramp[::-1]
        self.raw_averaged *= win

        t_hw_sp = generator.short_pulse_half_width(self.recording['f2']) * 1E-3
        bw = 2 / t_hw_sp
        df = int(0.5*bw)

        cutoff = fdp + np.array([-df, df],)
        self.dpoae_signal = filters.bp_pass_filter(
            self.raw_averaged,
            filters.BP_ORDER,
            samplerate,
            cutoff
        )

    def process_raw_data(
        self,
        recorded_signal: npt.NDArray[np.float64],
        block_size: int,
    ) -> npt.NDArray[np.float64]:
        """Processes recorded signal to obtain an optimized average.

        Args:
            recorded_signal: float array of measurement data
            block_size: Size of each recording block in samples

        Returns:
            Array of floats containing the averaged signal.
        """

        # Obtain an integer number of recorded blocks
        total_blocks = int(len(recorded_signal)/block_size)
        block_data = recorded_signal[:total_blocks*block_size]
        blocks = block_data.reshape(-1, block_size)

        # perform PTPV averaging
        num_ensembles = int(total_blocks / generator.NUM_PTPV_SEGMENTS)
        ensembles_size = (num_ensembles, block_size)
        ensembles = np.zeros(ensembles_size, dtype=np.float32)
        for i in range(num_ensembles):
            idx_start = i * generator.NUM_PTPV_SEGMENTS
            idx_stop = (i + 1) * generator.NUM_PTPV_SEGMENTS
            ensembles[i, :] = np.mean(blocks[idx_start:idx_stop], axis=0)

        ensemble_power = np.zeros(num_ensembles)
        for i in range(num_ensembles):
            ensemble_power[i] = processing.estimate_power(ensembles[i,:])

        self.averager.setup(num_ensembles)
        self.averager.noise_values = ensemble_power.astype(np.float32)
        self.averager.i_received = num_ensembles
        self.averager.evaluate_averaging()


        if len(self.averager.accepted_idx):
            avg = ensembles[self.averager.accepted_idx, :].mean(axis=0)
            self.log.info(
                'Optimized averaging: accepted blocks %d/%d.',
                self.averager.stats.num_accepted_blocks,
                self.averager.i_received
            )
        else:
            self.log.error('Optimized averaging failed.')
            avg = ensembles.mean(axis=0)

        return avg

    def save_data(self, file_name: str) -> None:
        """Saves data to json."""
        if self.recording is None:
            return
        samplerate = self.recording['samplerate']
        t = np.arange(self.recording['num_block_samples']) / samplerate * 1E3
        d = {
            'f1': self.recording['f1'],
            'f2': self.recording['f2'],
            'level1': self.recording['level1'],
            'level2': self.recording['level2'],
            'samplerate': samplerate,
            't': t.tolist(),
            'y': self.dpoae_signal.tolist()
        }
        files.save_result_to_json(file_name + '.json', d)
