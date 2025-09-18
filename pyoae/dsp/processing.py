"""Python script to process and visualize a single recording."""

from pathlib import Path
from typing import TypedDict, cast, NotRequired

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
# from matplotlib.figure import Figure
# from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
import scipy.signal as sig


from pyoae import files
from pyoae import generator
from pyoae.calib import MicroTransferFunction
from pyoae.dsp.opt_avg import OptAverage


NUM_PTPV_SEGMENTS = 4
HP_ORDER = 1201
BP_ORDER = 1201
RAMP_DURATION = 2


class PulseDpoaeMsrmtData(TypedDict):
    """Raw measurement data of a pulsed DPOAE recording."""
    recorded_signal: npt.NDArray[np.float32]
    samplerate: float
    f1: float
    f2: float
    level1: float
    level2: float
    num_block_samples: int
    recorded_sync: npt.NDArray[np.float32]
    average: NotRequired[npt.NDArray[np.float64] | None]


def load_recording(file_path: str | Path) -> PulseDpoaeMsrmtData | None:
    """Loads the raw measurement data from binary file."""
    try:
        data = np.load(file_path)
    except FileNotFoundError as e:
        print(e)
        return None

    if 'average' in data:
        # for backwards compatibility
        avg = data['average']
    else:
        avg = None

    return {
        'recorded_signal': data['recorded_signal'],
        'samplerate': float(data['samplerate']),
        'f1': float(data['f1']),
        'f2': float(data['f2']),
        'level1': float(data['level1']),
        'level2': float(data['level2']),
        'num_block_samples': int(data['num_block_samples']),
        'recorded_sync': data['recorded_sync'],
        'average': avg
    }


def estimate_energy(y: npt.NDArray[np.float32]) -> float:
    """Estimates the signal energy as RMS for a given signal."""
    return np.sqrt(np.mean(np.square(y)))


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
    # reflect-pad only at the *start* to reduce startup transient
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
    # reflect-pad only at the *start* to reduce startup transient
    y_ext = np.pad(y.astype(np.float64, copy=False), (0, D), mode="edge")
    y_f = cast(np.ndarray, sig.lfilter(b, 1.0, y_ext))
    return y_f[D:D + y.shape[0]]


class PulseDpoaeProcessor:
    """Instance to process a pulsed DPOAE recording."""

    recording: PulseDpoaeMsrmtData | None

    averager: OptAverage

    filtered_recording: npt.NDArray[np.float64]

    raw_averaged: npt.NDArray[np.float64]

    dpoae_signal: npt.NDArray[np.float64]

    mic_trans_fun: MicroTransferFunction | None

    def __init__(
        self,
        recording: PulseDpoaeMsrmtData | None = None,
        mic: MicroTransferFunction | None = None,
        file_path: str | Path | None = None,
        mic_path: str | Path | None = None
    ) -> None:
        """Initialize processor and load recording."""
        if file_path:
            self.recording = load_recording(file_path)
        else:
            self.recording = recording
        if self.recording is None:
            return
        self.averager = OptAverage()
        self.filtered_recording = np.empty(0, dtype=np.float64)
        self.raw_averaged = np.empty(0, dtype=np.float64)
        self.dpoae_signal = np.empty(0, dtype=np.float64)
        if mic_path:
            mic_calib_data = files.load_micro_calib(mic_path)
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
            # TODO: log warning
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
        bw = 0.5 * (1 / t_hw_sp)
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

        ensemble_energy = np.zeros(num_ensembles)
        for i in range(num_ensembles):
            ensemble_energy[i] = estimate_energy(ensembles[i,:])

        self.averager.setup(num_ensembles)
        self.averager.noise_values = ensemble_energy.astype(np.float32)
        self.averager.i_received = num_ensembles
        self.averager.evaluate_averaging()


        if len(self.averager.accepted_idx):
            avg = ensembles[self.averager.accepted_idx, :].mean(axis=0)
        else:
            print('Optimized averaging failed.')
            avg = ensembles.mean(axis=0)

        return avg

    def plot(self) -> None:
        """Plots the data."""
        if self.recording is None:
            return

        samplerate = self.recording['samplerate']
        fig, axes = plt.subplots(3, 1, figsize=(10, 6))
        axes: list[Axes]

        # plot recording overview
        t_rec = np.arange(len(self.filtered_recording)) / samplerate
        axes[0].plot(t_rec, self.filtered_recording, linewidth=0.5)

        t_avg = np.arange(self.recording['num_block_samples']) / samplerate * 1E3
        axes[1].plot(t_avg, self.raw_averaged, linewidth=0.5)
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
            f'L1: {self.recording['level1']} dB SPL, '
            f'L2: {self.recording['level2']} dB SPL, '
            f'f2: {self.recording['f2']} Hz'
        )
        axes[1].set_title(
            f'Accepted blocks: {self.averager.stats.num_accepted_blocks}'''
            f'/{self.averager.i_received}')

        fig.tight_layout()
        plt.show()

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
