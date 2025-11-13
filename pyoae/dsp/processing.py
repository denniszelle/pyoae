"""Module with classes and functions to process and visualize recordings."""

from logging import Logger
from pathlib import Path
from typing import TypedDict, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
# from matplotlib.figure import Figure
# from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import scipy.signal as sig

from pyoae import files
from pyoae import generator
from pyoae import get_logger
from pyoae.calib import MicroTransferFunction
from pyoae.dsp.opt_avg import OptAverage


NUM_PTPV_SEGMENTS = 4
HP_ORDER = 1201
BP_ORDER = 1201
RAMP_DURATION = 2


class DpoaeMsrmtData(TypedDict):
    """A general container for raw DPOAE measurement data."""
    recorded_signal: npt.NDArray[np.float32]
    samplerate: float
    f1: float
    f2: float
    level1: float
    level2: float
    num_block_samples: int
    recorded_sync: npt.NDArray[np.float32]


class ContDpoaeRecording(TypedDict):
    """Structured content from a DPOAE recording file."""
    recording: DpoaeMsrmtData
    average: npt.NDArray[np.float64] | None
    spectrum: npt.NDArray[np.float64] | None


class PulseDpoaeRecording(TypedDict):
    """Structured content from a pulsed DPOAE recording file."""
    recording: DpoaeMsrmtData
    average: npt.NDArray[np.float64] | None
    signal: npt.NDArray[np.float64] | None


def _msrmt_to_cont_recording(
    msrmt_data: DpoaeMsrmtData
) -> ContDpoaeRecording:
    """Creates a continuous DPOAE recording from measurement data."""
    return {
        'recording': msrmt_data,
        'average': None,
        'spectrum': None
    }

def _msrmt_to_pulse_recording(
    msrmt_data: DpoaeMsrmtData
) -> PulseDpoaeRecording:
    """Creates a pulsed DPOAE recording dictionary from measurement data."""
    return {
        'recording': msrmt_data,
        'average': None,
        'signal': None
    }

def estimate_power(y: npt.NDArray[np.float32 | np.float64]) -> float:
    """Estimates the signal power as RMS for a given signal."""
    return np.sqrt(np.mean(np.square(y)))


def estimate_spectral_noise(
    y: npt.NDArray[np.floating],
    num_samples: int,
    f_signal: float,
    samplerate: float,
    noise_bin_offset: int = 10,
    num_noise_bins: int = 10
) -> float:
    """Estimate narrow-band noise around a harmonic's bin as RMS amplitude.

    Returns:
        Noise RMS (same units as y's amplitude; per-bin, not per Hz).
    """
    # Detrend (remove DC) to reduce leakage into neighbors
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()

    # FFT (one-sided), explicit size
    Y = np.fft.rfft(y, n=num_samples)

    # Convert to one-sided **peak** amplitude per bin
    # A_peak = 2*|Y|/N for 0<k<N/2; DC and Nyquist are not doubled
    mag = np.abs(Y) / num_samples
    if num_samples % 2 == 0:
        # even N: rfft has N/2+1 bins, last is Nyquist
        mag[1:-1] *= 2.0
    else:
        # odd N: last bin is not Nyquist; all bins except DC are doubled
        mag[1:] *= 2.0

    # Convert to **RMS** per bin (sine RMS = peak / sqrt(2))
    mag_rms = mag / np.sqrt(2.0)

    # Locate signal bin (coherent sampling assumed)
    # Since we're ensuring that continuous primary tones
    # exhibit an integer number of periods within an
    # acquisition block, we're save to use the following
    # formula.
    idx_signal = int(round(f_signal * num_samples / samplerate))
    # Alternatively search for the closest bin using the fft
    # frequencies:
    # frequencies = np.fft.rfftfreq(num_samples, d=1/samplerate)
    # idx_signal = np.argmin(np.abs(frequencies - f_signal))

    # Build noise-bin indices on both sides, skipping close bins
    idx_left_start = idx_signal - noise_bin_offset - num_noise_bins
    idx_left_end = idx_signal - noise_bin_offset
    idx_right_start = idx_signal + noise_bin_offset
    idx_right_end = idx_signal + noise_bin_offset + num_noise_bins

    # Clip to valid range [0, len(mag_rms)-1] and drop empty sides gracefully
    n_bins = mag_rms.shape[0]
    left = np.arange(max(0, idx_left_start),  max(0, idx_left_end))
    right = np.arange(min(n_bins, idx_right_start), min(n_bins, idx_right_end))
    noise_bins = np.concatenate((left, right))
    if noise_bins.size == 0:
        raise ValueError("No valid noise bins (signal too close to edges).")

    # Power averaging for noise, then RMS
    noise_rms = float(np.sqrt(np.mean(mag_rms[noise_bins] ** 2)))

    return noise_rms


def high_pass_filter(
    y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    num_taps: int,
    samplerate: float,
    cutoff_hz: float = 200.0,
) -> npt.NDArray[np.float64]:
    """FIR high-pass with causal phase; same length as input."""
    if num_taps % 2 == 0:
        # HP linear-phase FIRs should use odd taps for Type-I symmetry
        num_taps += 1
    b = sig.firwin(num_taps, cutoff_hz, pass_zero="highpass", fs=samplerate)  # type: ignore

    D = (num_taps - 1) // 2  # group delay
    y_ext = np.pad(y.astype(np.float64, copy=False), (0, D), mode="edge")
    y_f = cast(np.ndarray, sig.lfilter(b, 1.0, y_ext))
    return y_f[D:D + y.shape[0]]


def bp_pass_filter(
    y: npt.NDArray[np.float64],
    num_taps: int,
    samplerate: float,
    cutoff_hz: npt.NDArray,
) -> npt.NDArray[np.float64]:
    """FIR high-pass with causal phase; same length as input."""
    if num_taps % 2 == 0:
        # HP linear-phase FIRs should use odd taps for Type-I symmetry
        num_taps += 1
    b = sig.firwin(num_taps, cutoff_hz, pass_zero="bandpass", fs=samplerate)  # type: ignore

    D = (num_taps - 1) // 2  # group delay
    y_ext = np.pad(y.astype(np.float64, copy=False), (0, D), mode="edge")
    y_f = cast(np.ndarray, sig.lfilter(b, 1.0, y_ext))
    return y_f[D:D + y.shape[0]]


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

        axes[0].set_xlim(0, t_rec[-1])
        axes[1].set_xlim(0, t_avg[-1])
        axes[2].set_xlim(0, t_avg[-1])
        axes[0].set_xlabel("Recording Time (s)")
        axes[1].set_ylabel('Amp. (full scale)')
        axes[2].set_ylabel('p (muPa)')
        axes[2].set_xlabel('t (ms)')

        # rec_lim = axes[0].get_ylim()
        # axes[1].set_ylim(rec_lim)
        axes[0].set_title(
            f'L1: {self.recording["level1"]} dB SPL, '
            f'L2: {self.recording["level2"]} dB SPL, '
            f'f2: {self.recording["f2"]} Hz'
        )
        axes[1].set_title('Raw Average')
        axes[2].set_title('Filtered Average - DPOAE Signal')
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
        self.filtered_recording = high_pass_filter(
            recorded_signal,
            HP_ORDER,
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
            RAMP_DURATION * 1E-3 * samplerate
        )
        ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
        ramp = ramp.astype(np.float32)
        win = np.ones_like(self.raw_averaged)
        win[:ramp_len] = ramp
        win[-ramp_len:] = ramp[::-1]
        self.raw_averaged *= win

        t_hw_sp = generator.short_pulse_half_width(self.recording['f2']) * 1E-3
        bw = 1 / t_hw_sp
        df = int(0.5*bw)

        cutoff = fdp + np.array([-df, df],)
        self.dpoae_signal = bp_pass_filter(
            self.raw_averaged,
            BP_ORDER,
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
        num_ensembles = int(total_blocks/NUM_PTPV_SEGMENTS)
        ensembles_size = (num_ensembles, block_size)
        ensembles = np.zeros(ensembles_size, dtype=np.float32)
        for i in range(num_ensembles):
            idx_start = i * NUM_PTPV_SEGMENTS
            idx_stop = (i + 1) * NUM_PTPV_SEGMENTS
            ensembles[i, :] = np.mean(blocks[idx_start:idx_stop], axis=0)

        ensemble_power = np.zeros(num_ensembles)
        for i in range(num_ensembles):
            ensemble_power[i] = estimate_power(ensembles[i,:])

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
        self.filtered_recording = high_pass_filter(
            recorded_signal,
            HP_ORDER,
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
            block_noise[i] = estimate_spectral_noise(
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
