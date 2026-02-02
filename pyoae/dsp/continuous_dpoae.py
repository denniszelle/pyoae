"""Module with classes and functions to process and visualize recordings."""

from logging import Logger
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt

from pyoae import files
from pyoae import get_logger
from pyoae.calib import MicroTransferFunction
from pyoae.dsp import filters
from pyoae.dsp import processing
from pyoae.dsp.containers import (
    DpoaeMsrmtData,
    ContDpoaeRecording,
)
from pyoae.dsp.opt_avg import OptAverage


def _msrmt_to_cont_recording(
    msrmt_data: DpoaeMsrmtData
) -> ContDpoaeRecording:
    """Creates a continuous DPOAE recording from measurement data."""
    return {
        'recording': msrmt_data,
        'average': None,
        'spectrum': None
    }


class ContDpoaeResult:
    """Instance to manage a DPOAE result from continuous recording."""

    log: Logger

    recording: DpoaeMsrmtData

    raw_averaged: npt.NDArray[np.float64]

    dpoae_spectrum: npt.NDArray[np.float64]

    def __init__(self, cont_recording: ContDpoaeRecording) -> None:
        self.log = get_logger(__class__.__name__)
        self.recording = cont_recording['recording']
        if cont_recording['average'] is None:
            self.raw_averaged = np.empty(0, dtype=np.float64)
        else:
            self.raw_averaged = cont_recording['average']
        if cont_recording['spectrum'] is None:
            self.dpoae_spectrum = np.empty(0, dtype=np.float64)
        else:
            self.dpoae_spectrum = cont_recording['spectrum']

    def plot(self, block_loop: bool = True) -> None:
        """Plots the result data."""
        if self.recording is None:
            return

        samplerate = self.recording['samplerate']
        num_block_samples = self.recording['num_block_samples']
        num_recording_samples = len(self.recording['recorded_signal'])

        fig, axes = plt.subplots(3, 1, figsize=(10, 6))
        axes: list[Axes]

        # plot recording overview
        t_rec = np.arange(num_recording_samples) / samplerate
        axes[0].plot(t_rec, self.recording['recorded_signal'], linewidth=0.5)
        # axes[0].plot(t_rec, self.filtered_recording, linewidth=0.5)
        # axes[0].plot([0.1], [0], 'rx')

        frequencies = np.fft.rfftfreq(num_block_samples, 1 / samplerate)
        t_avg = np.arange(num_block_samples) / samplerate * 1E3
        if self.raw_averaged.size:
            axes[1].plot(t_avg, self.raw_averaged, linewidth=0.5)
        if self.dpoae_spectrum.size:
            axes[2].plot(frequencies, self.dpoae_spectrum, linewidth=0.5)

        fdp = 2 * self.recording['f1'] - self.recording['f2']

        f_min = np.floor((fdp - 100) / 1000) * 1000
        f_max = np.ceil((self.recording['f2'] + 100) / 1000) * 1000

        axes[0].set_xlim(0, t_rec[-1])
        axes[1].set_xlim(0, t_avg[-1])
        axes[2].set_xlim(max(200, f_min), min(f_max, frequencies[-1]))
        axes[0].set_xlabel("Recording Time (s)")
        axes[1].set_ylabel('Amp. (full scale)')
        axes[2].set_ylabel('L (dB SPL)')
        axes[2].set_xlabel('f (Hz)')
        # axes[2].set_xscale('log')

        # rec_lim = axes[0].get_ylim()
        # axes[1].set_ylim(rec_lim)
        axes[0].set_title(
            f'L1: {self.recording["level1"]} dB SPL, '
            f'L2: {self.recording["level2"]} dB SPL, '
            f'f2: {self.recording["f2"]} Hz'
        )
        axes[1].set_title('DPOAE Spectrum')
        fig.tight_layout()
        plt.show(block=block_loop)


class ContDpoaeProcessor(ContDpoaeResult):
    """Instance to process a continuous DPOAE recording."""

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
        cont_recording = _msrmt_to_cont_recording(msrmt_data)
        super().__init__(cont_recording)
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
        num_block_samples = self.recording['num_block_samples']
        recorded_signal -= np.mean(recorded_signal)  # remove DC
        # perform high-pass filtering
        self.filtered_recording = filters.high_pass_filter(
            recorded_signal,
            filters.scale_filter_order(filters.HP_ORDER, samplerate),
            samplerate
        )

        fdp = 2 * self.recording['f1'] - self.recording['f2']
        self.raw_averaged = self.process_raw_data(
            self.filtered_recording,
            num_block_samples,
            fdp,
            samplerate
        )

        # compute spectrum
        spectrum = 2*np.abs(np.fft.rfft(self.raw_averaged)) / num_block_samples
        # dBFS and dB SPL represent RMS values
        # assume FFT bins represent sine waves and estimate
        # RMS by dividing by sqrt(2)
        spectrum /= np.sqrt(2)
        if self.mic_trans_fun is None:
            spectrum = 20 * np.log10(spectrum)
        else:
            spectrum /= self.mic_trans_fun.amplitudes
            spectrum = 20 * np.log10(spectrum/20)
        self.dpoae_spectrum = spectrum

    def process_raw_data(
        self,
        recorded_signal: npt.NDArray[np.float64],
        block_size: int,
        fdp: float,
        samplerate: float
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
        # remove first and last block due to onset and offset ramps
        blocks = blocks[1:-1,:]
        # perform PTPV averaging
        num_blocks = total_blocks - 2
        block_noise = np.zeros(num_blocks)
        for i in range(num_blocks):
            block_noise[i] = processing.estimate_spectral_noise(
                blocks[i,:],
                block_size,
                fdp,
                samplerate
            )

        self.averager.setup(num_blocks)
        self.averager.noise_values = block_noise.astype(np.float32)
        self.averager.i_received = num_blocks
        self.averager.evaluate_averaging()

        if len(self.averager.accepted_idx):
            avg = blocks[self.averager.accepted_idx, :].mean(axis=0)
            self.log.info(
                'Optimized averaging: accepted blocks %d/%d.',
                self.averager.stats.num_accepted_blocks,
                self.averager.i_received
            )
        else:
            self.log.error('Optimized averaging failed.')
            avg = blocks.mean(axis=0)

        return avg

    def save_data(self, file_name: str) -> None:
        """Saves data to json."""
        if self.recording is None:
            return
        samplerate = self.recording['samplerate']
        num_block_samples = self.recording['num_block_samples']
        t = np.arange(num_block_samples) / samplerate * 1E3
        frequencies = np.fft.rfftfreq(num_block_samples, 1 / samplerate)
        d = {
            'f1': self.recording['f1'],
            'f2': self.recording['f2'],
            'level1': self.recording['level1'],
            'level2': self.recording['level2'],
            'samplerate': samplerate,
            't': t.tolist(),
            'y': self.raw_averaged.tolist(),
            'f': frequencies,
            'spectrum': self.dpoae_spectrum
        }
        files.save_result_to_json(file_name + '.json', d)
