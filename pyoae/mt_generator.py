"""Module that contains classes and functions for multitone generation."""

from dataclasses import dataclass
from logging import Logger

import numpy as np
import numpy.typing as npt

from pyoae import get_logger
from pyoae import generator
from pyoae.device.device_config import DeviceConfig
from pyoae.calib_storage import MicroTransferFunction
from pyoae import protocols


def ramp_envelope(n_samples: int, ramp_samples: int) -> np.ndarray:
    """Generate mask to multiply ramps to start and end of a signal."""
    env = np.ones(n_samples)

    if ramp_samples > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_samples)))
        env[:ramp_samples] = ramp
        env[-ramp_samples:] = ramp[::-1]

    return env


@dataclass
class MultiToneResult:
    """Class to define results for multitone measurements."""

    spectra: list[npt.NDArray[np.float32]]
    """List of spectra for each multitone cluster"""

    freq_spectra: npt.NDArray[np.float32]
    """Frequency values of spectra"""

    frequencies: npt.NDArray[np.float32]
    """Frequencies of the multitones"""

    freq_idc: npt.NDArray[np.int32]
    """Indices of frequencies"""

    amplitude: npt.NDArray[np.float32]
    """Amplitudes at multitone frequencies corrected by input amplitude"""

    raw_amplitude: npt.NDArray[np.float32]
    """Amplitudes of multitones without input correction"""

    phase: npt.NDArray[np.float32]
    """Phase at multitone frequencies"""


class MultiToneDefinition:
    """Class for multitone definition and generation"""

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
        frequencies: npt.NDArray[np.float32],
        phases: npt.NDArray[np.float32],
        amplitudes: npt.NDArray[np.float32],
        cluster_idc: npt.NDArray[np.int32],
        log: Logger | None = None
        ):

        self.logger = log or get_logger()
        self.frequencies = frequencies
        self.phases = phases
        self.amplitudes = amplitudes
        self.cluster_idc = cluster_idc

    def get_unique_cluster_indices(self) -> npt.NDArray[np.int32]:
        """Return the unique cluster indices"""
        return np.unique(self.cluster_idc)

    def get_cluster_indices(self) -> npt.NDArray[np.int32]:
        """Return the unique cluster indices"""
        return np.unique(self.cluster_idc)

    def get_n_clusters(self):
        """Return the number of clusters for the multitone definition."""
        return len(self.get_unique_cluster_indices())

    def get_frequencies(self) -> npt.NDArray[np.float32]:
        """Return the frequencies of the spectral tones."""
        return self.frequencies

    def get_phases(self) -> npt.NDArray[np.float32]:
        """Return the phases of the spectral tones."""
        return self.phases

    def get_cluster_signals(self, cluster_idx: int):
        """Return frequencies, amplitudes and phases for one cluster."""
        mask = self.cluster_idc == cluster_idx

        frequencies = self.frequencies[mask]
        phases = self.phases[mask]
        amplitudes = self.amplitudes[mask]

        return frequencies, amplitudes, phases

    def generate_mt_signal(
        self,
        num_samples: int,
        sample_rate: float,
        ramp_duration: float
    ) -> npt.NDArray[np.float32]:

        """Create a multi-tone signal used for output calibration."""

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

            self.ramp_correction = 1/np.mean(env**2)

        return signal

    def _samples_per_cluster(self, n_samples_recorded) -> int:
        """Get the number of samples per cluster"""
        n_clusters = self.get_n_clusters()
        return n_samples_recorded // n_clusters



class MultiToneAnalyzer:
    """Analyze a recorded multitone signal relative to its definition."""

    mt_definition: MultiToneDefinition
    """Definition of the multitone set"""

    recorded_signal: npt.NDArray[np.float32]
    """Recorded signal"""

    sample_rate: float
    """Sampling rate of the system"""

    def __init__(
        self,
        mt_definition: MultiToneDefinition,
        recorded_signal: npt.NDArray[np.float32],
        sample_rate: float
    ):
        self.mt = mt_definition
        self.recorded_signal = recorded_signal
        self.sample_rate = sample_rate

    def _samples_per_cluster(self):
        """Return the number of samples assigned to a cluster"""
        n_clusters = self.mt.get_n_clusters()
        return len(self.recorded_signal) // n_clusters

    def _get_cluster_segment(self, cluster_idx: int):
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
        micro_tf: MicroTransferFunction | None
    ):
        """Compute spectrum of a signal and apply a transfer function"""
        n = len(segment)

        ramp_samples = int(
            DeviceConfig.ramp_duration
            * 1E-3
            * DeviceConfig.sample_rate
        )
        ramp = ramp_envelope(n, ramp_samples)
        windowed = segment * ramp

        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        spectrum = 2* np.fft.rfft(windowed) / len(segment)
        np.divide(spectrum, np.sqrt(2), spectrum)
        spectrum = spectrum.astype(np.complex64)

        if micro_tf is not None:
            spectrum /= micro_tf.get_interp_transfer_function(freqs)

        return freqs, spectrum

    def compute_result(
        self, micro_tf: MicroTransferFunction | None
    ) -> MultiToneResult:
        """Compute result of the measured signal relative the generated one"""
        cluster_indices = sorted(self.mt.get_unique_cluster_indices())
        spectra = []
        segments = []

        # Collect all frequencies from the definition
        all_frequencies = []
        all_freq_idc = []
        all_amplitudes = []
        all_raw_amplitudes = []
        all_phases = []
        freqs = np.ndarray(0,np.float32)

        for cluster_idx in cluster_indices:
            segment = self._get_cluster_segment(cluster_idx)
            freqs, spectrum = self._compute_spectrum(segment, micro_tf)
            spectra.append(np.abs(spectrum).astype(np.float32))
            segments.append(segment)

            cluster_freqs, cluster_amps, cluster_phases = (
                self.mt.get_cluster_signals(
                    cluster_idx
                )
            )

            freq_idc = np.argmin(
                np.abs(freqs[:, None] - cluster_freqs[None, :]),
                axis=0
            )

            raw_amps = np.abs(spectrum[freq_idc])
            if self.mt.ramp_correction is None:
                amps = raw_amps/cluster_amps
            else:
                amps = raw_amps/cluster_amps*self.mt.ramp_correction
            phases = np.angle(spectrum[freq_idc]) - cluster_phases

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
            phase=np.unwrap(all_phases[arg_sort])
        )


def compute_mt_frequencies(
    f_start: float,
    f_stop: float,
    lines_per_octave: float,
    df: float,
    extra_density: float = 0.0,
) -> np.ndarray:
    """Computes multi-tone frequencies.

    Computes frequency lines from start to stop frequency adjusted to
    segment length with the specified lines per octave.

    """

    freqs = [f_start]
    f = f_start

    while f < f_stop:
        # linear increase of density over frequency
        t = (f - f_start) / (f_stop - f_start)
        current_lpo = lines_per_octave + extra_density * t

        b = 2 ** (1 / current_lpo)
        f = f * b
        f = np.round(f / df) * df

        if f > freqs[-1]:   # prevent duplicates due to rounding
            freqs.append(f)

    return np.array(freqs)


def compute_mt_phases(num_frequencies: int) -> npt.NDArray[np.floating]:
    """Computes approximately equally distributed phases"""
    phi = np.zeros(num_frequencies)
    for i in range(num_frequencies):
        phi[i] = np.random.uniform(0, 2 * np.pi)
    return phi

def generate_mt_def(msrmt_params: protocols.CalibMsrmtParams
) -> protocols.CalibMsrmtDef:
    """Generate multitone definition"""
    df = 1/msrmt_params['block_duration']*msrmt_params['num_clusters']
    mt_frequencies = compute_mt_frequencies(
        msrmt_params['f_start'],
        msrmt_params['f_stop'],
        msrmt_params['lines_per_octave'],
        df,
        extra_density=20.0
    )
    # Remove redundant frequencies
    mt_frequencies = np.unique(mt_frequencies)
    num_mt_frequencies = len(mt_frequencies)
    mt_phases = compute_mt_phases(num_mt_frequencies)
    mt_amplitudes = (
        np.ones_like(mt_frequencies)
        * msrmt_params['amplitude_per_line']
    ).astype(np.float32)
    cluster_idc = np.arange(num_mt_frequencies, dtype=np.int32)
    cluster_idc = cluster_idc % msrmt_params['num_clusters']

    calib_def: protocols.CalibMsrmtDef = {
        'block_duration': msrmt_params['block_duration'],
        'num_averaging_blocks': msrmt_params['num_averaging_blocks'],
        'frequencies': mt_frequencies,
        'phases': mt_phases,
        'amplitudes': mt_amplitudes,
        'cluster_idc': cluster_idc
    }
    return calib_def
