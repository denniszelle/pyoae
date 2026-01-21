"""Functions to generate output signals."""

from dataclasses import dataclass
from logging import Logger
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import windows

from pyoae import converter
from pyoae import get_logger
from pyoae.calib import OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.protocols import DpoaeMsrmtParams, PulseDpoaeMsrmtParams, PulseStimulus
# from pyoae.signals import PeriodicRampSignal


SYNC_CROSS: Final[int] = 10
"""Number of minimum zero crossings in sync signal."""

SYNC_AMPLITUDE: Final[float] = 0.2
"""Amplitude of the sync signal in digital full scale."""

SYNC_FREQUENCY: Final[float] = 4000
"""Carrier frequency of the sync pulse in Hz."""

NUM_PTPV_SEGMENTS = 4
"""Number of segments for a full PTPV ensemble."""

PRIMARY1_PTPV_SHIFT = np.pi / (NUM_PTPV_SEGMENTS / 2)
"""Phase shift of low-frequency primary.

Phase shifts for default of 4 segments:
  f1 -> 90°, f2 -> 180°
"""

PRIMARY2_PTPV_SHIFT = np.pi / (NUM_PTPV_SEGMENTS / 4)
"""Phase shift of high-frequency primary.

For default number of 4 segments, f2 short pulses have
a 180° phase shift.
"""


logger = get_logger()


def short_pulse_half_width(f2: float) -> float:
    """Calculate the half-width of a short stimulus pulse.

    Returns:
        half-width in ms.
    """
    return max(13071.3/f2, 13071.3/4000)


def create_pulse_mask(
    block_duration: float,
    pulse: PulseStimulus,
    t_hw_sp: float
) -> PulseStimulus:
    """Creates the mask for a pulsed primary tone."""
    t_on = pulse['t_on'] * 1E-3
    if pulse['is_short_pulse']:
        # scale duration and ramps as short pulse
        pulse_hw = pulse['duration'] - 0.5 * (pulse['t_fall'] + pulse['t_rise'])
        pulse_scale = t_hw_sp / pulse_hw
        t_rise = pulse['t_rise'] * 1E-3 * pulse_scale
        t_fall = pulse['t_fall'] * 1E-3 * pulse_scale
        duration = pulse['duration'] * 1E-3 * pulse_scale
    else:
        t_rise = pulse['t_rise'] * 1E-3
        t_fall = pulse['t_fall'] * 1E-3
        duration = pulse['duration'] * 1E-3

    # perform some basic sanity checks
    duration = min(block_duration-t_on, duration)
    if t_rise + t_fall > duration:
        t_rise = 0.5 * duration
        t_fall = 0.5 * duration

    return {
        't_rise': t_rise,
        't_fall': t_fall,
        't_on': t_on,
        'duration': duration,
        'is_short_pulse': pulse['is_short_pulse']
    }


def create_pulse_pattern(
        pulse: PulseStimulus,
        f: float,
        phi: float
    ) -> npt.NDArray[np.float32]:
    """Creates a pulse pattern with unity amplitude."""
    num_samples = int(pulse['duration'] * DeviceConfig.sample_rate)
    t = np.arange(num_samples, dtype=np.float32) / DeviceConfig.sample_rate
    win = np.ones(num_samples, dtype=np.float32)

    num_r_samples = int(pulse['t_rise'] * DeviceConfig.sample_rate)
    rising_ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(num_r_samples)/(2*num_r_samples)))
    rising_ramp = rising_ramp.astype(np.float32)

    num_f_samples = int(pulse['t_fall'] * DeviceConfig.sample_rate)
    falling_ramp = 0.5*(1 + np.cos(2*np.pi*np.arange(num_f_samples)/(2*num_f_samples)))
    falling_ramp = falling_ramp.astype(np.float32)

    win[:num_r_samples] = rising_ramp
    win[-num_f_samples:] = falling_ramp
    y = np.sin(2*np.pi*f*t + phi).astype(np.float32)
    return y * win


def create_ptpv_signals(
    pulse_mask: PulseStimulus,
    frequency: float,
    amplitude: float,
    phase_shift: float,
    num_block_samples: int,
    num_segments: int = 4
) -> list[npt.NDArray[np.float32]]:
    """Creates a list with PTPV signals."""
    # Generate output signals
    stimuli: list[npt.NDArray[np.float32]] = []

    idx_on = int(pulse_mask['t_on'] * DeviceConfig.sample_rate)
    num_pulse_samples = int(pulse_mask['duration'] * DeviceConfig.sample_rate)

    # create stimuli with phase shift
    for i in range(num_segments):
        pulse_pattern = create_pulse_pattern(
            pulse_mask,
            frequency,
            i*phase_shift
        )
        pulse = amplitude * pulse_pattern
        # move to appropriate position in signal template
        signal_template = np.zeros(num_block_samples, dtype=np.float32)
        signal_template[idx_on:idx_on+num_pulse_samples] = pulse
        stimuli.append(signal_template)
    return stimuli


@dataclass
class DpoaeStimulus:
    """Container for DPOAE stimulus parameters."""

    f1: float
    """Frequency of first primary tone in Hz."""

    f2: float
    """Frequency of second primary tone in Hz."""

    level1: float
    """Stimulus level of first primary tone in dB SPL."""

    level2: float
    """Stimulus level of second primary tone in dB SPL."""


class ContDpoaeStimulus(DpoaeStimulus):
    """Container for continuous DPOAE primary tones."""

    def calculate_frequencies(
        self,
        msrmt_params: DpoaeMsrmtParams,
        block_duration: float
    ) -> None:
        """Calculate primary-tone frequencies f1 and f2 for cDPOAE acquisition."""
        self.f2 = correct_frequency(msrmt_params['f2'], block_duration)
        if msrmt_params['f1'] is None:
            if msrmt_params['f2f1_ratio'] is None:
                self.f1 = 0.0  # invalid stimulus parameters
                # TODO: log error
            else:
                self.f1 = self.f2/msrmt_params['f2f1_ratio']
                self.f1 = correct_frequency(self.f1, block_duration)
        else:
            self.f1 = correct_frequency(msrmt_params['f1'], block_duration)

        logger.info('Setting primary-tone frequencies:')
        logger.info('  f1: %.2f Hz | f2: %.2f Hz', self.f1, self.f2)
        logger.info('  f2/f1 = %.3f', self.f2/self.f1)

    def generate_stimuli(
        self,
        num_block_samples: int,
        output_calibration: OutputCalibration | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Generates primary tones for continuous DPOAE acquisition."""

        if output_calibration is None:
            # No calibration for output channels available.
            amplitude1, amplitude2 = calculate_full_scale_amplitudes(
                self.level2,
                self.level1
            )
        else:
            pressure1, pressure2 = calculate_pressure_amplitudes(
                self.level2,
                self.level1
            )
            amplitude1 = output_calibration.pressure_to_full_scale(
                0,
                pressure1,
                self.f1
            )
            amplitude2 = output_calibration.pressure_to_full_scale(
                1,
                pressure2,
                self.f2
            )
            logger.info('Setting output amplitudes for DPOAE acquisition:')
            logger.info('p1: %.1f muPa (%.6f re FS).', pressure1, amplitude1)
            logger.info('p2: %.1f muPa (%.6f re FS).', pressure2, amplitude2)

        # Verify output amplitudes
        amplitude1 = check_output_limit(amplitude1)
        amplitude2 = check_output_limit(amplitude2)
        # Generate output signals
        samples = np.arange(num_block_samples, dtype=np.float32)
        t = samples / DeviceConfig.sample_rate
        stimulus1 = amplitude1 * np.sin(2*np.pi*self.f1*t).astype(np.float32)
        stimulus2 = amplitude2 * np.sin(2*np.pi*self.f2*t).astype(np.float32)
        return (stimulus1, stimulus2)


class PulseDpoaeStimulus(DpoaeStimulus):
    """Container for continuous DPOAE primary tones."""

    f1_pulse_mask: PulseStimulus | None = None

    f2_pulse_mask: PulseStimulus | None = None

    logger: Logger

    def calculate_frequencies(
        self,
        msrmt_params: PulseDpoaeMsrmtParams
    ) -> None:
        """Calculate primary-tone frequencies f1 and f2 for pulsed DPOAE acquisition."""
        self.logger = get_logger()
        self.f2 = msrmt_params['f2']
        if msrmt_params['f1'] is None:
            if msrmt_params['f2f1_ratio'] is None:
                self.f1 = 0.0  # invalid stimulus parameters
                # TODO: log error
            else:
                self.f1 = self.f2/msrmt_params['f2f1_ratio']
        else:
            self.f1 = msrmt_params['f1']

        self.logger.info('Setting primary-tone frequencies:')
        self.logger.info('  f1: %.2f Hz | f2: %.2f Hz', self.f1, self.f2)
        self.logger.info('  f2/f1 = %.3f', self.f2/self.f1)

    def create_stimulus_mask(
        self,
        block_duration: float,
        msrmt_params: PulseDpoaeMsrmtParams
    ) -> None:
        """Creates the stimulus masks for both primary-tone pulses."""
        t_hw_sp = short_pulse_half_width(msrmt_params['f2'])
        # create f1 pulse markers in seconds
        f1_pulse = msrmt_params['f1_pulse']
        f2_pulse = msrmt_params['f2_pulse']
        self.f1_pulse_mask = create_pulse_mask(
            block_duration,
            f1_pulse,
            t_hw_sp
        )
        self.f2_pulse_mask = create_pulse_mask(
            block_duration,
            f2_pulse,
            t_hw_sp
        )
        self.logger.debug('f1 pulse mask: %s.', self.f1_pulse_mask)
        self.logger.debug('f2 pulse mask: %s.', self.f2_pulse_mask)

    def generate_stimuli(
        self,
        num_block_samples: int,
        output_calibration: OutputCalibration | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Generates primary tones for continuous DPOAE acquisition."""

        if output_calibration is None:
            # No calibration for output channels available.
            amplitude1, amplitude2 = calculate_full_scale_amplitudes(
                self.level2,
                self.level1
            )
        else:
            pressure1, pressure2 = calculate_pressure_amplitudes(
                self.level2,
                self.level1
            )
            amplitude1 = output_calibration.pressure_to_full_scale(
                0,
                pressure1,
                self.f1
            )
            amplitude2 = output_calibration.pressure_to_full_scale(
                1,
                pressure2,
                self.f2
            )
            logger.info('Setting output amplitudes for DPOAE acquisition:')
            logger.info('p1: %.1f muPa (%.6f re FS).', pressure1, amplitude1)
            logger.info('p2: %.1f muPa (%.6f re FS).', pressure2, amplitude2)

        if self.f1_pulse_mask is None or self.f2_pulse_mask is None:
            # TODO: Consider raising ValueError
            return (np.zeros(0, np.float32), np.zeros(0, np.float32))

        # Verify output amplitudes
        amplitude1 = check_output_limit(amplitude1)
        amplitude2 = check_output_limit(amplitude2)

        f1_stimuli = create_ptpv_signals(
            self.f1_pulse_mask,
            self.f1,
            amplitude1,
            PRIMARY1_PTPV_SHIFT,
            num_block_samples,
            num_segments=NUM_PTPV_SEGMENTS
        )
        f2_stimuli = create_ptpv_signals(
            self.f2_pulse_mask,
            self.f2,
            amplitude2,
            PRIMARY2_PTPV_SHIFT,
            num_block_samples,
            num_segments=NUM_PTPV_SEGMENTS
        )

        # concatenate PTPV segments to create the stimulus signals
        stimulus1 = np.concatenate(f1_stimuli).astype(np.float32, copy=False)
        stimulus2 = np.concatenate(f2_stimuli).astype(np.float32, copy=False)

        return stimulus1, stimulus2


def correct_frequency(frequency: float, block_duration: float) -> float:
    """Corrects frequency to obtain an integer number of periods.

    Args:
        frequency: Stimulus frequency in Hz.
        block_duration: Length of acquisition block in seconds.
          A block represents a time segment that is repeatedly
          presented and used for averaging.
    """
    periods = block_duration*frequency
    periods = round(periods)
    return periods/block_duration


def calculate_pt1_level(msrmt_params: DpoaeMsrmtParams) -> float:
    """Calculates the stimulus level of the first primary tone."""
    if msrmt_params['level1'] is None:
        # as first approximation, use Kummer et al. 1998
        # TODO: add other rules
        #level1 = 0.4 * msrmt_params['level2'] + 39
        # Kempa et al. 2025
        level1 = 0.5 * msrmt_params['level2'] + 35
    else:
        level1 = msrmt_params['level1']
    return level1


def calculate_full_scale_amplitudes(
    level2: float, level1: float | None
) -> tuple[float, float]:
    """Computes digital full-scale amplitudes from dBFS levels.

    Uses levels as dBFS (0 dBFS = digital full scale, i.e. 1).
    Applies equal stimulus level rule (L1 = L2) if L1
    was not specified
    """
    level1 = level1 or level2

    # check boundaries
    if level1 > 0:
        logger.warning(
            'Value %.2f dBFS for output level 1 exceeds maximum of 0 dBFS.',
            level1
        )
        # use an arbitrary save fallback value
        level1 = -20.0
        logger.warning(
            'Using fallback output level %.2f dBFS instead.',
            level1
        )
        logger.warning('Please check protocol and calibration files.')

    if level2 > 0:
        logger.warning(
            'Value %.2f dBFS for output level 2 exceeds maximum of 0 dBFS.',
            level2
        )
        # use an arbitrary save fallback value
        level2 = -20.0
        logger.warning(
            'Using fallback output level %.2f dBFS instead.',
            level2
        )
        logger.warning('Please check protocol and calibration files.')


    amplitude1 = converter.db_to_lin(level1)
    amplitude2 = converter.db_to_lin(level2)
    amplitude1 = max(0.0, min(amplitude1, 1.0))
    amplitude2 = max(0.0, min(amplitude2, 1.0))
    logger.info('Calculating digital full-scale amplitudes for output.')
    logger.info('  L1: %.2f dBFS -> .%6f re FS.', level1, amplitude1)
    logger.info('  L2: %.2f dBFS -> .%6f re FS.', level2, amplitude2)

    return (amplitude1, amplitude2)


def calculate_pressure_amplitudes(
    level2: float,
    level1: float
) -> tuple[float, float]:
    """Calculates peak pressure amplitudes from dB SPL levels."""
    amplitude1 = converter.db_spl_to_peak_mupa(level1)
    amplitude2 = converter.db_spl_to_peak_mupa(level2)
    logger.info(
        'Calculating peak output pressures from sound pressure levels:'
    )
    logger.info('  L1: %.1f dB SPL -> %.2f muPa', level1, amplitude1)
    logger.info('  L2: %.1f dB SPL -> %.2f muPa', level2, amplitude2)
    return (amplitude1, amplitude2)


def check_output_limit(peak_amplitude: float) -> float:
    """Verifies and limits the maximum digital output."""
    if peak_amplitude > DeviceConfig.max_digital_output:
        logger.warning(
            'Output amplitude %.5f exceeds maximum of %.2f re FS.',
            peak_amplitude,
            DeviceConfig.max_digital_output
        )
        logger.warning(
            'Output amplitude set to %.2f re FS.',
            DeviceConfig.max_digital_output
        )
        return DeviceConfig.max_digital_output
    return peak_amplitude


def generate_sync(sample_rate: float) -> npt.NDArray[np.float32]:
    """Generates and returns the sync signal.

    Args:
        fs: Sampling frequency in Hz

    Returns:
        sync_pulse: 1D array of sync pulse signal
    """
    k = np.ceil(sample_rate / (4.0 * SYNC_FREQUENCY))
    samples_per_sync_period = 4.0 * k
    f_sync = sample_rate / samples_per_sync_period
    p = np.ceil((SYNC_CROSS + 1) / 2)
    t_off = (p / f_sync) - 1/sample_rate
    num_samples = int(t_off * sample_rate)
    t = np.arange(num_samples) / sample_rate
    y = np.sin(2 * np.pi * f_sync * t)
    w = windows.tukey(num_samples)
    sync_pulse = SYNC_AMPLITUDE * w * y
    return sync_pulse


def compute_mt_frequencies(
    f_start: float,
    f_stop: float,
    lines_per_octave: float
) -> npt.NDArray[np.floating]:
    """Computes multi-tone frequencies.

    Computes frequency lines from start to stop frequency adjusted to
    segment length with the specified lines per octave.
    """
    b = 2 ** (1/lines_per_octave)
    n = int(np.log(f_stop/f_start)/np.log(b))
    f = f_start * b ** (np.arange(n))
    f = np.round(f)
    return f


def compute_mt_phases(num_frequencies: int) -> npt.NDArray[np.floating]:
    """Computes approximately equally distributed phases"""
    phi = np.zeros(num_frequencies)
    for i in range(num_frequencies):
        phi[i] = np.random.uniform(0, 2 * np.pi)
    return phi


def interlace_mt_frequencies(
    frequencies: npt.NDArray[np.floating],
    num_tones: int
) -> list[npt.NDArray[np.floating]]:
    """Creates interlaced frequency lists.

    Splits frequency array into `num_tones` interlaced sets.
    Each set contains every nth frequency starting at a different offset.
    """
    return [frequencies[i::num_tones] for i in range(num_tones)]


def get_time_vector(
    num_samples: int,
    sample_rate: float
) -> npt.NDArray[np.float32]:
    """Computes and returns a time vector in seconds."""
    time_vec = np.arange(num_samples) / sample_rate
    return time_vec.astype(np.float32)


def generate_mt_signal(
    num_samples: int,
    sample_rate: float,
    frequencies: npt.NDArray[np.float32],
    phases: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Create a multi-tone signal used for output calibration."""
    time_vec = get_time_vector(num_samples, sample_rate)

    mt_signal = np.zeros_like(time_vec)
    for i, f in enumerate(frequencies):
        mt_signal += np.sin(2 * np.pi * f * time_vec + phases[i])

    return mt_signal
