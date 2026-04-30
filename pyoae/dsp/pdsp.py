"""Module with classes and functions to process and visualize recordings."""

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import scipy.signal as sig

from pyoae import files
from pyoae import generator
from pyoae import get_logger
from pyoae.calib_transfer import EarSimTransferFunction, MicroTransferFunction
from pyoae.dsp import averaging
from pyoae.dsp import filters
from pyoae.dsp import noise
from pyoae.dsp import spectral
from pyoae.dsp.averaging import AveragingStrategy
from pyoae.dsp.containers import (
    DpoaeMsrmtData,
    PulseDpoaeRecording
)
from pyoae.dsp.filters import BpFilterOptions, FilterOptions
from pyoae.dsp.opt_avg import OptAverage


def _msrmt_to_pulse_recording(
    msrmt_data: DpoaeMsrmtData
) -> PulseDpoaeRecording:
    """Creates a pulsed DPOAE recording dictionary from measurement data."""
    return {
        'recording': msrmt_data,
        'average': None,
        'signal': None
    }


@dataclass
class StimulusTimeMarkers:
    """Time markers of the evoking pulsed stimulus."""

    t_on: float
    """Time stamp of the start of the stimulus pulse"""

    pulse_duration: float
    """Total duration of the stimulus pulse"""

    t_rise: float
    """Duration of the rising ramp of the stimulus"""

    t_fall: float
    """Duration of the falling ramp of the stimulus"""

    pulse_hw: float
    """Pulse half-width"""

    sp_rise_interval: npt.NDArray[np.float64]
    """Time stamps of the rising ramp interval"""

    sp_ss_interval: npt.NDArray[np.float64]
    """Time stamps of the steady-state interval"""

    sp_fall_interval: npt.NDArray[np.float64]
    """Time stamps of the falling ramp interval"""


@dataclass
class PulseDpoaeProcessOptions:
    """Options for the processing of pulse DPOAEs."""

    high_pass_options: FilterOptions

    band_pass_options: BpFilterOptions

    averaging_strategy: AveragingStrategy

    noise_options: noise.PulseDpoaeNoiseOptions


class PulseDpoaeResult:
    """Instance to manage a DPOAE result from pulsed recording."""

    log: Logger

    time_markers: StimulusTimeMarkers | None

    recording: DpoaeMsrmtData

    raw_averaged: npt.NDArray[np.float64]

    dpoae_signal: npt.NDArray[np.float64]

    dpoae_envelope: npt.NDArray[np.float64]

    dpoae_phase: npt.NDArray[np.float64]

    def __init__(self, pulsed_recording: PulseDpoaeRecording) -> None:
        self.log = get_logger(__class__.__name__)
        self.time_markers = None

        self.recording = pulsed_recording['recording']
        if pulsed_recording['average'] is None:
            self.raw_averaged = np.empty(0, dtype=np.float64)
        else:
            self.raw_averaged = pulsed_recording['average']
        if pulsed_recording['signal'] is None:
            self.dpoae_signal = np.empty(0, dtype=np.float64)
        else:
            self.dpoae_signal = pulsed_recording['signal']

    def get_fdp(self) -> float:
        """Returns the DPOAE frequency."""
        return 2 * self.recording['f1'] - self.recording['f2']

    def get_frequency_ratio(self) -> float:
        """Returns the stimulus-frequency ratio."""
        return self.recording['f2'] / self.recording['f1']

    def set_analytic_results(self) -> None:
        """Derives envelope and instantaneous phase form analytic signal."""
        if self.dpoae_signal is None:
            self.log.error(
                'Failed to calculate analytic signal. '
                'No filtered DPOAE signal available.'
            )
            return
        analytic_signal = cast(np.ndarray, sig.hilbert(self.dpoae_signal))
        self.dpoae_envelope = np.abs(analytic_signal)
        phi_raw = np.angle(analytic_signal)
        fdp = self.get_fdp()
        samplerate = self.recording['samplerate']
        t = np.arange(self.recording['num_block_samples']) / samplerate
        self.dpoae_phase = np.unwrap(phi_raw) - 2 * np.pi * fdp * t

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
            f'f2/f1: {self.get_frequency_ratio()}'
        )
        axes[1].set_title('Raw Average')
        dpoae_max = np.max(self.dpoae_signal)
        axes[2].set_title(
            f'Filtered Average - DPOAE Signal (Max: {dpoae_max:.2f})'
        )
        fig.tight_layout()
        plt.show(block=block_loop)

    def plot_envelope(
        self,
        block_loop: bool = True,
        show_stimulus: bool = True,
        show_title: bool = True,
        data_color: tuple[float, float, float] = (0.3, 0.3, 0.3),
        stimulus_color: tuple[float, float, float] = (0.9, 0.1, 0.1)
    ) -> None:
        """Plots the envelope and instantaneous phase of the pulsed DPOAE data."""
        if self.dpoae_envelope is None or self.dpoae_phase is None:
            return

        samplerate = self.recording['samplerate']
        fig, axes = plt.subplots(
            2, 1,
            figsize=(10, 6),
            gridspec_kw={'height_ratios': [2, 1]}
        )
        axes: list[Axes]
        ax_signal = axes[0]
        ax_phase = axes[1]

        t = np.arange(self.recording['num_block_samples']) / samplerate * 1E3

        ax_signal.plot(
            t,
            self.dpoae_signal,
            linewidth=0.5,
            color=data_color
        )
        ax_signal.plot(
            t,
            self.dpoae_envelope,
            linewidth=1.0,
            color=data_color
        )

        ax_signal.set_ylabel('p (muPa)')
        y_max = np.max(np.abs(self.dpoae_signal))
        y_max = max(np.ceil(y_max / 50), 3) * 50

        if show_stimulus and self.time_markers is not None:
            ax_signal.plot(
                self.time_markers.sp_rise_interval,
                np.array([-y_max+5, -y_max+5]),
                color=stimulus_color,
                linestyle='--'
            )
            ax_signal.plot(
                self.time_markers.sp_ss_interval,
                np.array([-y_max+5, -y_max+5]),
                color=stimulus_color
            )
            ax_signal.plot(
                self.time_markers.sp_fall_interval,
                np.array([-y_max+5, -y_max+5]),
                color=stimulus_color,
                linestyle='--'
            )
        ax_signal.set_xlim(t[0], t[-1])
        ax_signal.set_ylim(-y_max, y_max)

        ax_phase.plot(
            t,
            self.dpoae_phase,
            linewidth=1.0,
            color=data_color
        )
        ax_phase.set_xlabel('t (ms)')
        ax_phase.set_ylabel('phi (rad)')
        ax_phase.set_xlim(t[0], t[-1])

        if show_title:
            fig.suptitle(
                f'L1={self.recording["level1"]} dB SPL,'
                f'L2={self.recording["level1"]} dB SPL,'
                f'\nf2/f1={self.get_frequency_ratio()}, '
                f'fdp: {self.get_fdp():.1f} Hz', fontsize=10
            )

        fig.tight_layout()
        plt.show(block=block_loop)


class PulseDpoaeProcessor(PulseDpoaeResult):
    """Instance to process a pulsed DPOAE recording."""

    averager: OptAverage

    filtered_recording: npt.NDArray[np.float64]

    mic_trans_fun: MicroTransferFunction | None

    ear_sim_trans_fun: EarSimTransferFunction | None

    options: PulseDpoaeProcessOptions

    def __init__(
        self,
        msrmt_data: DpoaeMsrmtData,
        mic: MicroTransferFunction | None = None,
        ear_sim_tf: EarSimTransferFunction | None = None,
        mic_path: str | Path | None = None,
        ear_sim_path: str | Path | None = None,
    ) -> None:
        """Initialize processor and load recording."""
        pulsed_recording = _msrmt_to_pulse_recording(msrmt_data)
        super().__init__(pulsed_recording)

        high_pass_options = filters.default_high_pass_options(
            self.recording['samplerate']
        )
        band_pass_options = filters.default_band_pass_options(
            self.recording['samplerate'],
            self.get_fdp(),
            self.recording['f2']
        )
        noise_options = noise.default_pdpoae_noise_options(
            self.recording['samplerate'],
            self.get_fdp(),
            self.recording['f2']
        )

        self.options = PulseDpoaeProcessOptions(
            high_pass_options=high_pass_options,
            band_pass_options=band_pass_options,
            averaging_strategy=AveragingStrategy.ENSEMBLE,
            noise_options=noise_options
        )

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

        if ear_sim_path:
            ear_sim_data = files.load_ear_sim_calib(ear_sim_path)
            if ear_sim_data is not None:
                self.ear_sim_trans_fun = EarSimTransferFunction(
                    ear_sim_data['transfer_function']
                )
        else:
            self.ear_sim_trans_fun = ear_sim_tf

    def process_msrmt(self) -> None:
        """Process measurement to extract DPOAE."""
        if self.recording is None:
            return

        self.prepare_recording()
        self.raw_averaged = self.average_raw_data()
        self.apply_input_calibration()

        bp_options = self.options.band_pass_options
        if bp_options['enable']:
            self.dpoae_signal = filters.bp_pass_filter(
                self.raw_averaged,
                bp_options['num_taps'],
                self.recording['samplerate'],
                bp_options['cutoff_hz'],
                ramp_size=bp_options['ramp_size']
            )
        else:
            self.dpoae_signal = np.empty(0, dtype=np.float64)

    def prepare_recording(self) -> None:
        """Performs basic signal conditioning of raw recording."""
        samplerate = self.recording['samplerate']
        recorded_signal = self.recording['recorded_signal']
        recorded_signal -= np.mean(recorded_signal)  # remove DC

        hp_options = self.options.high_pass_options
        if hp_options['enable']:
            # perform high-pass filtering
            self.filtered_recording = filters.high_pass_filter(
                recorded_signal,
                hp_options['num_taps'],
                samplerate,
                cutoff_hz=hp_options['cutoff_hz']
            )
        else:
            self.filtered_recording = recorded_signal.astype(np.float64)

    def average_raw_data(self) -> npt.NDArray[np.float64]:
        """Processes recorded signal to obtain an optimized average.

        Args:
            recorded_signal: float array of measurement data
            block_size: Size of each recording block in samples

        Returns:
            Array of floats containing the averaged signal.
        """
        num_block_samples = self.recording['num_block_samples']
        if self.options.averaging_strategy is AveragingStrategy.ENSEMBLE:
            blocks = averaging.calculate_ptpv_ensembles(
                self.filtered_recording,
                num_block_samples,
                generator.NUM_PTPV_SEGMENTS
            )
        else:
            raise NotImplementedError(
                'BLOCK Averaging Strategy is not implemented yet.'
            )

        num_blocks = blocks.shape[0]

        noise_options = self.options.noise_options
        # estimate noise in spectral domain for each block
        cplx_spectra = spectral.block_cplx_spectrum(
            blocks,
            ramp_size=noise_options['ramp_size']
        )
        amp_spectra = spectral.block_abs_spectrum(cplx_spectra, num_block_samples)

        block_noise = noise.batch_pdpoae_spectral_noise(
            amp_spectra,
            num_block_samples,
            noise_options['f_signal'],
            noise_options['signal_bw'],
            self.recording['samplerate'],
            noise_options['num_noise_bins']
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

    def apply_input_calibration(self) -> None:
        """Applies the microphone calibration."""
        if self.mic_trans_fun is None:
            self.log.warning(
                'Microphone data missing. Falling back to unity conversion.'
            )
            return

        samplerate = self.recording['samplerate']
        raw_spec = np.fft.rfft(self.raw_averaged)
        raw_spec_frequencies = np.fft.rfftfreq(
            len(self.raw_averaged),
            1 / samplerate
        )
        mic_tf = self.mic_trans_fun.get_interp_transfer_function(
            raw_spec_frequencies
        )
        raw_spec /= mic_tf
        if self.ear_sim_trans_fun is not None:
            ear_sim_tf = self.ear_sim_trans_fun.get_interp_transfer_function(
                raw_spec_frequencies
            )
            raw_spec *= ear_sim_tf

        self.raw_averaged = np.real(np.fft.irfft(raw_spec))

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
