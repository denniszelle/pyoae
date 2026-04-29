"""Module that contains classes and functions for multi-tone generation."""

from dataclasses import dataclass
from logging import Logger

import numpy as np
import numpy.typing as npt

from pyoae import get_logger
from pyoae import generator
from pyoae.device.device_config import DeviceConfig
from pyoae.dsp.spectral import cplx_spectrum, get_spec_frequenies
from pyoae.calib_transfer import EarSimTransferFunction, MicroTransferFunction
from pyoae import protocols
from pyoae.protocols import CalibMsrmtDef, CalibMsrmtParams


def ramp_envelope(n_samples: int, ramp_samples: int) -> np.ndarray:
    """Generate a signal envelope with cosine ramps at start and end.

    This function creates an array of length `n_samples` where the first and last
    `ramp_samples` are shaped with a half-cosine ramp. The central portion remains
    at unity.

    Args:
        n_samples: Total number of samples in the signal.
        ramp_samples: Number of samples used for the rising and falling ramps.

    Returns:
        A numpy array of shape `(n_samples,)` containing the envelope values ranging
        from 0 to 1, with smooth ramps at the edges.
    """
    env = np.ones(n_samples)

    if ramp_samples > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_samples)))
        env[:ramp_samples] = ramp
        env[-ramp_samples:] = ramp[::-1]

    return env


@dataclass
class MultiToneResult:
    """Class to define results for multi-tone measurements."""

    spectra: list[npt.NDArray[np.float32]]
    """List of spectra for each multi-tone cluster"""

    freq_spectra: npt.NDArray[np.float32]
    """Frequency values of spectra"""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the multi-tones"""

    freq_idc: npt.NDArray[np.int32]
    """Indices of frequencies"""

    amplitude: npt.NDArray[np.float32]
    """Amplitudes at multi-tone frequencies corrected by input amplitude"""

    raw_amplitude: npt.NDArray[np.float32]
    """Amplitudes of multi-tones without input correction"""

    phase: npt.NDArray[np.float32]
    """Phase at multi-tone frequencies"""


class MultiToneDefinition:
    """Class for multi-tone definition and generation"""

    logger: Logger
    """Class logger for debug, info, warning and error messages"""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the spectral signals"""

    phases: npt.NDArray[np.float32]
    """Phases of the spectral signals"""

    amplitudes: npt.NDArray[np.float32]
    """Amplitudes of the spectral signal"""

    cluster_idc: npt.NDArray[np.int32]
    """Cluster indices of the spectral signals"""

    ramp_correction: np.float32 | None = None
    """Ramp correction to correct amplitudes"""

    def __init__(
        self,
        msrmt_params: CalibMsrmtDef | CalibMsrmtParams,
        log: Logger | None = None,
    ) -> None:

        self.logger = log or get_logger()

        if 'num_clusters' in msrmt_params:
            calib_def = generate_mt_def(msrmt_params)
        elif 'frequencies' in msrmt_params:
            calib_def = msrmt_params
        else:
            self.logger.error('Invalid protocol type.')
            return False

        self.frequencies = calib_def['frequencies']
        self.phases = calib_def['phases']
        self.amplitudes = calib_def['amplitudes']
        self.cluster_idc = calib_def['cluster_idc']

    def get_unique_cluster_indices(self) -> npt.NDArray[np.int32]:
        """Return the unique cluster indices"""
        return np.unique(self.cluster_idc)

    def get_cluster_indices(self) -> npt.NDArray[np.int32]:
        """Return the unique cluster indices"""
        return np.unique(self.cluster_idc)

    def get_n_clusters(self):
        """Return the number of clusters for the multi-tone definition."""
        return len(self.get_unique_cluster_indices())

    def get_frequencies(self) -> npt.NDArray[np.float32]:
        """Return the frequencies of the spectral tones."""
        return self.frequencies

    def get_phases(self) -> npt.NDArray[np.float32]:
        """Return the phases of the spectral tones."""
        return self.phases

    def get_cluster_signals(self, cluster_idx: int) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Return frequencies, amplitudes and phases for one cluster."""
        mask = self.cluster_idc == cluster_idx

        frequencies = self.frequencies[mask]
        phases = self.phases[mask]
        amplitudes = self.amplitudes[mask]

        return frequencies, amplitudes, phases

    def generate_mt_signal(
        self, num_samples: int, sample_rate: float, ramp_duration: float
    ) -> npt.NDArray[np.float32]:
        """Generate a multi-tone signal for output calibration.

        This method synthesizes a multi-tone signal divided into clusters of frequencies.
        Each cluster is applied sequentially over the total number of samples, with a
        cosine ramp applied to the start and end of each cluster segment to avoid clicks.

        Args:
            num_samples: Total number of samples for the generated signal.
            sample_rate: Sampling rate in Hz.
            ramp_duration: Duration of the rising and falling ramps in milliseconds.

        Returns:
            A numpy array of shape `(num_samples,)` containing the multi-tone signal
            with ramps applied to each cluster.
        """

        time_vec = generator.get_time_vector(num_samples, sample_rate)
        signal = np.zeros_like(time_vec)

        cluster_indices = sorted(self.get_unique_cluster_indices())
        n_clusters = len(cluster_indices)

        samples_per_cluster = num_samples // n_clusters
        ramp_samples = int(ramp_duration * 1E-3 * sample_rate)

        for i, cluster_idx_i in enumerate(cluster_indices):
            start = i * samples_per_cluster
            if i < n_clusters - 1:
                end = start + samples_per_cluster
            else:
                end = num_samples

            t_segment = time_vec[: end - start]
            segment = np.zeros_like(t_segment)

            cluster_freqs, cluster_amps, cluster_phases = (
                self.get_cluster_signals(cluster_idx_i)
            )

            for i, freq_i in enumerate(cluster_freqs):

                segment += cluster_amps[i] * np.cos(
                    2 * np.pi * freq_i * t_segment + cluster_phases[i]
                )

            env = ramp_envelope(len(segment), ramp_samples)
            segment *= env
            signal[start:end] += segment
            self.ramp_correction = 1 / np.mean(env**2)

        return signal

    def _samples_per_cluster(self, n_samples_recorded) -> int:
        """Get the number of samples per cluster"""
        n_clusters = self.get_n_clusters()
        return n_samples_recorded // n_clusters


class MultiToneAnalyzer:
    """Analyze a recorded multi-tone signal relative to its definition."""

    mt_definition: MultiToneDefinition
    """Definition of the multi-tone set"""

    recorded_signal: npt.NDArray[np.float32]
    """Recorded signal"""

    sample_rate: float
    """Sampling rate of the system"""

    def __init__(
        self,
        mt_definition: MultiToneDefinition,
        recorded_signal: npt.NDArray[np.float32],
        sample_rate: float,
    ) -> None:
        self.mt = mt_definition
        self.recorded_signal = recorded_signal
        self.sample_rate = sample_rate

    def _samples_per_cluster(self):
        """Return the number of samples assigned to a cluster"""
        n_clusters = self.mt.get_n_clusters()
        return len(self.recorded_signal) // n_clusters

    def _get_cluster_segment(
        self, cluster_idx: int
    ) -> npt.NDArray[np.float32]:
        """Get signal segment assigned to a cluster"""
        clusters = sorted(self.mt.get_unique_cluster_indices())
        i = clusters.index(cluster_idx)
        n = self._samples_per_cluster()
        start = i * n
        # Last cluster takes the remainder
        end = start + n if i < len(clusters) - 1 else len(self.recorded_signal)
        return self.recorded_signal[start:end]

    def _compute_spectrum(
        self,
        segment: npt.NDArray[np.float32],
        micro_tf: MicroTransferFunction | None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.complex64]]:
        """Compute spectrum of a signal and apply a transfer function"""

        ramp_samples = int(
            DeviceConfig.ramp_duration * 1E-3 * DeviceConfig.sample_rate
        )

        freqs = get_spec_frequenies(len(segment), self.sample_rate)
        spectrum = cplx_spectrum(segment, ramp_samples)
        # Convert to RMS values
        np.divide(spectrum, np.sqrt(2), spectrum)

        if micro_tf is not None:
            spectrum /= micro_tf.get_interp_transfer_function(freqs)
        return freqs, spectrum

    def get_ear_sim_corrections(
        self,
        frequencies: npt.NDArray[np.float32],
        ear_sim_tf: EarSimTransferFunction,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return corrections of ear simulator transfer function"""
        cplx_tf = ear_sim_tf.get_interp_transfer_function(frequencies)
        return np.abs(cplx_tf), np.angle(cplx_tf)

    def compute_result(
        self,
        micro_tf: MicroTransferFunction | None,
        ear_sim_tf: EarSimTransferFunction | None,
    ) -> MultiToneResult:
        """Compute the measurement result relative to the generated signal.

        Args:
            micro_tf: Optional microphone transfer function used to correct the
                measured spectrum. If `None`, the raw measurement is used.

        Returns:
            MultiToneResult object that contains the calibration results.
        """
        cluster_indices = sorted(self.mt.get_unique_cluster_indices())
        spectra = []
        segments = []

        # Collect all frequencies from the definition
        all_frequencies = []
        all_freq_idc = []
        all_amplitudes = []
        all_raw_amplitudes = []
        all_phases = []
        freqs = np.ndarray(0, np.float32)

        for cluster_idx in cluster_indices:
            segment = self._get_cluster_segment(cluster_idx)
            freqs, spectrum = self._compute_spectrum(
                segment, micro_tf
            )
            spectra.append(spectrum.astype(np.complex64))
            segments.append(segment)

            cluster_freqs, cluster_amps, cluster_phases = (
                self.mt.get_cluster_signals(cluster_idx)
            )

            if ear_sim_tf is not None:
                ear_sim_amps, ear_sim_phases = self.get_ear_sim_corrections(
                    cluster_freqs, ear_sim_tf
                )
            else:
                ear_sim_amps = np.ones_like(cluster_amps)
                ear_sim_phases = np.zeros_like(cluster_phases)

            freq_idc = np.argmin(
                np.abs(freqs[:, None] - cluster_freqs[None, :]), axis=0
            )

            raw_amps = np.abs(spectrum[freq_idc])
            if self.mt.ramp_correction is None:
                amps = raw_amps / cluster_amps * ear_sim_amps
            else:
                amps = (
                    raw_amps
                    / cluster_amps
                    * ear_sim_amps
                    * self.mt.ramp_correction
                )
            phases = (
                np.angle(spectrum[freq_idc]) - cluster_phases + ear_sim_phases
            )

            all_frequencies = np.r_[all_frequencies, cluster_freqs]
            all_amplitudes = np.r_[all_amplitudes, amps]
            all_freq_idc = np.r_[all_freq_idc, freq_idc]
            all_raw_amplitudes = np.r_[all_raw_amplitudes, raw_amps]
            all_phases = np.r_[all_phases, phases]

        arg_sort = np.argsort(all_frequencies)

        return MultiToneResult(
            spectra=spectra,
            freq_spectra=freqs,
            frequencies=all_frequencies[arg_sort],
            freq_idc=all_freq_idc[arg_sort],
            amplitude=all_amplitudes[arg_sort],
            raw_amplitude=all_raw_amplitudes[arg_sort],
            phase=np.unwrap(all_phases[arg_sort]),
        )


def compute_mt_frequencies(
    f_start: float,
    f_stop: float,
    lines_per_octave: float,
    df: float,
    extra_density: float = 0.0,
) -> np.ndarray:
    """Compute multi-tone frequencies for calibration or stimulus signals.

    Generates a sequence of frequencies from `f_start` to `f_stop` with a
    logarithmic spacing determined by `lines_per_octave`. The spacing can
    gradually increase with `extra_density`. Frequencies are adjusted to
    align with the discrete frequency resolution `df`.

    Args:
        f_start: Start frequency in Hz.
        f_stop: Stop frequency in Hz.
        lines_per_octave: Number of frequency lines per octave.
        df: Frequency resolution step (Hz) to round the computed frequencies.
        extra_density: Optional linear increase in density over the frequency
            range (default is 0.0).

    Returns:
        np.ndarray: Array of frequencies in Hz, starting at `f_start` and
        ending at or just above `f_stop`, rounded to the nearest multiple of `df`.
    """

    freqs = [f_start]
    f = f_start

    while f < f_stop:
        # linear increase of density over frequency
        t = (f - f_start) / (f_stop - f_start)
        current_lpo = lines_per_octave + extra_density * t

        b = 2 ** (1 / current_lpo)
        f_next = f * b
        f_next = np.round(f_next / df) * df

        if f_next <= f:
            f_next = f + df

        f = f_next
        freqs.append(f)

    return np.array(freqs)


def compute_mt_phases(num_frequencies: int) -> npt.NDArray[np.floating]:
    """Generate random phases for multi-tone signals.

    Args:
        num_frequencies: Number of phase values to generate.

    Returns:
        npt.NDArray[np.floating]: Array of length `num_frequencies` with
        random phase values in radians, uniformly distributed between 0 and 2π.
    """
    phi = np.zeros(num_frequencies)
    for i in range(num_frequencies):
        phi[i] = np.random.uniform(0, 2 * np.pi)
    return phi


def generate_mt_def(
    msrmt_params: protocols.CalibMsrmtParams,
) -> protocols.CalibMsrmtDef:
    """Generate a multi-tone signal definition for calibration measurements.

    Computes the frequencies, phases, amplitudes, and cluster indices
    for a multi-tone calibration signal based on the given measurement
    parameters.

    Args:
        msrmt_params: Dictionary of type CalibMsrmtParams:

    Returns:
        Dictionary of type CalibMsrmtDef containing the multi-tone definition

    """
    df = 1 / msrmt_params['block_duration'] * msrmt_params['num_clusters']
    mt_frequencies = compute_mt_frequencies(
        msrmt_params['f_start'],
        msrmt_params['f_stop'],
        msrmt_params['lines_per_octave'],
        df,
        extra_density=20.0,
    )
    # Remove redundant frequencies
    mt_frequencies = np.unique(mt_frequencies)
    num_mt_frequencies = len(mt_frequencies)
    mt_phases = compute_mt_phases(num_mt_frequencies)
    mt_amplitudes = (
        np.ones_like(mt_frequencies) * msrmt_params['amplitude_per_line']
    ).astype(np.float32)
    cluster_idc = np.arange(num_mt_frequencies, dtype=np.int32)
    cluster_idc = cluster_idc % msrmt_params['num_clusters']

    calib_def: protocols.CalibMsrmtDef = {
        'block_duration': msrmt_params['block_duration'],
        'num_averaging_blocks': msrmt_params['num_averaging_blocks'],
        'frequencies': mt_frequencies,
        'phases': mt_phases,
        'amplitudes': mt_amplitudes,
        'cluster_idc': cluster_idc,
    }
    return calib_def
