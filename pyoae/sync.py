"""Functions and classes for a synchronized measurement."""

from dataclasses import dataclass
from enum import auto, Enum
from logging import Logger
from multiprocessing import shared_memory
import multiprocessing as mp
import sys
from time import sleep
from typing import Any, Generic, Literal, Final, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.signal
import sounddevice as sd

from pyoae import generator
from pyoae import get_logger
from pyoae.device.device_config import DeviceConfig
from pyoae.plotting import LivePlotProcess, MsrmtEvents
from pyoae.signals import Signal


SYNC_DURATION: Final[float] = 0.5
"""Duration of the sync signal (with mute) in seconds.

Will be adjusted to a correspond to a number of samples
that is a multiple of the device's buffer size.
"""

SYNC_MIN_SNR: Final[float] = 30.0
"""Minimum SNR for the recorded sync signal.

Values below that threshold will raise a warning.
"""

NUM_INIT_RUNS: Final[int] = 10
"""Number of initialization runs to clear device buffers.

A measurement callback by the sound device is considered
a run. Depending on the system configuration and device
properties, the input buffers might contain previously
collected data. During the initialization the program
runs a mute signal that clears these buffers,
especially for the sync(monitoring) channel.
"""

USE_EARLY_LATENCY_CORRECTION: Final[bool] = False
"""Flag enabling(True)/disabling(False) early latency monitoring.

This requires the sync-input channel to be a monitoring channel,
i.e., the sync-input channel must differ from the microphone
input channel.

By default, the microphone input channel is ch. 01. Hence,
the recommend input channel for syncing and monitoring
is ch. 02.

Some audio interfaces may exhibit a varying number of
input/output buffers leading to an invalid compensation
of the device latency between playback and recording.

This occurs, if the device (or its drivers) utilizes
more buffers during the synchronization than during the
main measurement.

When enabling the early latency detection, a simple
maximum-based signal detection is performed while
compensating the expected device latency.
"""


SignalT = TypeVar('SignalT', bound=Signal)


class MsrmtState(Enum):
    """State of the measurement.

    This enumeration defines the various states, PyOAE
    takes during the acquisition.
    """

    SETUP = auto()
    """Initial state when in measurement setup."""

    COMPUTING = auto()
    """Operations required to start the measurement.

    E.g., latency computation.
    """

    SYNCING = auto()
    """Sync pulse is currently played."""

    STARTING = auto()
    """Start output and compensate latency."""

    RECORDING = auto()
    """In main measurement."""

    END_RECORDING = auto()
    """Switch playback to mute signal and record remaining data."""

    FINISHING = auto()
    """Wrapping up the processing of the main measurement."""

    FINISHED = auto()
    """Main measurement completed."""

    CANCELED = auto()
    """Measurement has been canceled."""


@dataclass
class RecordingData:
    """Basic information about the recording."""

    fs: float
    """Sampling frequency of the measurement device in Hz."""

    msrmt_duration: float
    """Total duration of the main measurement in seconds."""

    msrmt_samples: int
    """Total number of samples of the main measurement."""

    num_block_samples: int

    block_size: int
    """Block size of the measurement device."""


def get_input_channels(
        output_channels: list[int]
    ):
    """Return a list of input channels"""

    input_channels = []
    for output_channel_i in output_channels:
        mapping = DeviceConfig.output_input_mapping
        for mapping_pair in mapping:
            if mapping_pair[0] == output_channel_i:
                input_channels.append(
                    mapping_pair[1]
                )
    return input_channels


@dataclass
class HardwareData:
    """Basic information about input and output hardware."""

    n_in_channels: int
    """Number of input channels."""

    n_out_channels: int
    """Number of output channels."""

    input_device: int | str
    """Numeric ID or name of the input device."""

    output_device: int | str
    """Numeric ID or name of the output device."""

    output_channels: list[int]

    input_channels: list[int]

    max_in_channels: int | None = None

    max_out_channels: int | None = None

    def get_stream_input_channels(self) -> int:
        """Return input_channels suitable for sd.Stream."""

        if self.max_in_channels is None:
            dev = sd.query_devices(self.input_device)
            self.max_in_channels = dev['max_input_channels'] # type: ignore

        if not isinstance(self.max_in_channels, int):
            return -1

        if sys.platform == 'win32':
            # Windows requires the exact device channel count
            return self.max_in_channels

        if sys.platform in ('darwin', 'linux'):
            # macOS and Linux allow fewer channels
            return self.n_in_channels

        return self.n_in_channels

    def get_stream_output_channels(self) -> int:
        """Return output_channels suitable for sd.Stream"""

        if self.max_out_channels is None:
            dev = sd.query_devices(self.output_device)
            self.max_out_channels = dev['max_output_channels'] # type: ignore

        if not isinstance(self.max_out_channels, int):
            return -1

        if sys.platform == 'win32':
            # Windows requires the exact device channel count
            return self.max_out_channels

        if sys.platform in ('darwin', 'linux'):
            # macOS and Linux allow fewer channels
            return self.max_out_channels

        return self.max_out_channels

    def get_unique_input_channels(self) -> list[int]:
        """Return list of unique input channels"""
        active_in_channels = list({
            b for a, b in DeviceConfig.output_input_mapping
            if a in self.output_channels
        })
        return active_in_channels

    def get_output_msrmt_channels(self, msrmt_idx) -> list[int]:
        """Return list of output channels for a given measurement index"""
        return self.output_channels[2*msrmt_idx:2*msrmt_idx+2]


@dataclass
class LiveMsrmtData(Generic[SignalT]):
    """Variable storage for the live measurement."""

    play_idx: int
    """Current index in output array to get playback frames."""

    record_idx: int
    """Current index to write frames in input array."""

    latency_samples: int
    """DAQ latency of the measurement device in samples.

    The latency of the data acquisition is determined by presenting
    a tone burst on the sync output channel and recording it via an
    electric connection (short circuit) on SYNC_IN_CHANNEL.

    By presenting a mute signal of length latency_samples,
    output and input in the main measurement are synchronized.
    """

    output_signals: list[SignalT]
    """List of input signal objects."""

    sync_output: npt.NDArray[np.float32]
    """1D NumPy array with sync output signal."""

    sync_recorded: npt.NDArray[np.float32]
    """1D NumPy array to store recorded sync signal."""

    latency_type: Literal['low', 'high']
    """Specifies the desired latency setting for the audio stream.

    - 'low': Prioritizes low latency, but may increase CPU usage
        and be more sensitive to buffer underruns.
    - 'high': Prioritizes stability over latency, providing a more robust
        measurement.

    For more detailed information, see the documentation of sounddevice.Stream
    """

    recorded_signal: list[npt.NDArray[np.float32]]
    """1D NumPy array with recorded main measurement."""


class SyncMsrmt(Generic[SignalT]):
    """Class to perform a synchronized OAE measurement."""

    state: MsrmtState
    """The current state of the measurement.

    Used to control the measurement sequence.
    """

    recording_data: RecordingData
    """Object containing information about the recording."""

    hardware_data: HardwareData
    """Object containing information about the hardware."""

    live_msrmt_data: LiveMsrmtData[SignalT]
    """Object containing variables and data for the live measurement."""

    plot_process: LivePlotProcess
    """Object that executes the plot process"""

    msrmt_events: MsrmtEvents
    """List of events used for interaction with plot process"""

    i_init_runs: int
    """Counter of initial measurement callbacks with mute signal.

    In order to clear the input and output buffers of the audio
    interface, during setup a number of measurement callbacks are
    "ignored" by playing a mute signal and neglecting the recorded
    data.

    When this counter reaches the predefined number of initial runs,
    set by `NUM_INIT_RUNS` the measurement state switches to SYNCING.
    """

    shm: list[shared_memory.SharedMemory]
    """List of shared memory objects for recorded signal"""

    monitoring_amp: npt.NDArray[np.float32]
    """Signal amplitude of monitoring channel during playback start."""

    logger: Logger
    """Class logger for debug, info, warning, and error messages."""

    def __init__(
        self,
        recording_data: RecordingData,
        hardware_data: HardwareData,
        output_signals: list[SignalT],
        block_duration: float,
        latency_type: Literal['low', 'high']='high',
        log: Logger | None = None
    ) -> None:
        """Initializes the object.

        Args:
            recording_data: Provides meta data for the recording.
            hardware_data: Provides meta data for the used hardware.
        	output_signals: List of output signal objects or 1D output
              signal arrays
            latency_type: Latency type used('low' or 'high')
            log: Logger instance used for logging.
        """
        self.logger = log or get_logger()
        self.set_state(MsrmtState.SETUP)
        self.recording_data = recording_data
        self.hardware_data = hardware_data

        # compute duration of synchronization in multiples of block size
        k = (
            int(self.recording_data.fs * SYNC_DURATION)
            // recording_data.block_size
        )
        num_sync_samples = recording_data.block_size * k
        sync_duration = (num_sync_samples / recording_data.fs) * 1E3
        self.logger.debug(
            'Duration of sync-signal with mute: %.2f ms (%d samples)',
            sync_duration,
            num_sync_samples
        )

        sync_output = np.zeros(
            num_sync_samples,
            dtype=np.float32
        )
        sync_recorded = np.zeros(
            num_sync_samples,
            dtype=np.float32
        )
        sync_pulse = generator.generate_sync(self.recording_data.fs)
        sync_output[:len(sync_pulse)] = sync_pulse


        self.shm = []
        recorded_signal = []
        for i in range(hardware_data.get_stream_input_channels()-1):
            if i in hardware_data.get_unique_input_channels():
            # Only setup shared memory for channels that are used.
                self.shm.append(
                        shared_memory.SharedMemory(
                        create=True,
                        size=self.recording_data.msrmt_samples * np.float32().nbytes
                    )
                )
                recorded_signal.append(np.ndarray(
                        (self.recording_data.msrmt_samples,),
                        dtype=np.float32,
                        buffer=self.shm[-1].buf
                    )
                )

        self.record_idx_share = mp.Value('i', 0)

        self.block_duration = block_duration

        self.live_msrmt_data = LiveMsrmtData[SignalT](
            play_idx=0,
            record_idx=0,
            latency_samples=-1,
            output_signals=output_signals,
            sync_output=sync_output,
            sync_recorded=sync_recorded,
            latency_type=latency_type,
            recorded_signal=recorded_signal
        )
        self.i_init_runs = 0
        self.monitoring_amp = np.empty(0, dtype=np.float32)
        self.msrmt_events = MsrmtEvents(mp.Event(), mp.Event())

    def set_state(self, state: MsrmtState) -> None:
        """Sets the currently active state of this measurement."""
        self.state = state
        self.logger.debug("Measurement state set to %s.", self.state)

    def compute_latency(self) -> None:
        """Computes the stream latency from sync measurement."""

        # Perform some basic sanity checks
        sync_max_pos = np.argmax(np.abs(self.live_msrmt_data.sync_recorded))
        sync_max = np.abs(self.live_msrmt_data.sync_recorded[sync_max_pos])

        k = max(500, sync_max_pos-100)

        noise = np.sqrt(
            np.mean(np.square(self.live_msrmt_data.sync_recorded[:k]))
        )
        if noise == 0:
            # pylint: disable=no-member
            noise += np.finfo(np.float64).eps
            # pylint: enable=no-member
            self.logger.warning('Zero-noise detected.')

        # Very simple SNR estimator
        snr_ratio = (sync_max/np.sqrt(2))/noise
        if snr_ratio > 0:
            snr = 20 * np.log10(snr_ratio)
        else:
            self.logger.error(
                'Invalid value for SNR computation: %.2f', snr_ratio
            )
            snr = 0

        if snr < SYNC_MIN_SNR:
            self.logger.warning(
                'SNR of SYNC pulse below typical values: %.2f dB',
                snr
            )
            self.logger.warning(
                'Synchronization might be invalid. '
                'Measurement results can be corrupted.'
            )
        else:
            self.logger.info('SNR of SYNC pulse: %.2f dB', snr)

        self.live_msrmt_data.sync_recorded /= sync_max
        self.live_msrmt_data.sync_output /= np.max(
            np.abs(self.live_msrmt_data.sync_output)
        )
        correlation = scipy.signal.correlate(
            self.live_msrmt_data.sync_recorded,
            self.live_msrmt_data.sync_output
        )
        num_acq_samples = len(self.live_msrmt_data.sync_output)
        lag = np.argmax(correlation) - num_acq_samples + 1
        latency_time = (lag/self.recording_data.fs) * 1E3

        self.logger.info(
            'Measured latency: %.4f ms (%d samples).',
            latency_time,
            lag
        )
        self.logger.info(
            '%d samples acquired for latency determination.',
            num_acq_samples
        )
        if not isinstance(lag, np.integer):
            self.logger.error(
                'Cannot obtain latency. Invalid type: %s.',
                type(lag)
            )
            self.live_msrmt_data.latency_samples = 0
            return

        if lag < 0:
            self.logger.error(
                'Cannot obtain latency. Negative value detected.'
                'Please check cable connections.'
            )
        self.live_msrmt_data.latency_samples = max(int(lag), 0)

    def get_recorded_signal(self, input_channel) -> npt.NDArray[np.float32]:
        """Returns the recorded signal.

        Returns:
            Recorded signal with the length of already recorded data

        """
        if input_channel not in self.hardware_data.get_unique_input_channels():
            self.logger.error(
                'Invalid input channel for get_recorded_signal. \n'
                f'{input_channel} was given but options are only'
                f'{self.hardware_data.get_unique_input_channels()}')
            return np.zeros(0,np.float32)
        idx = self.hardware_data.get_unique_input_channels().index(input_channel)
        if self.state in [MsrmtState.RECORDING, MsrmtState.END_RECORDING]:
            return self.live_msrmt_data.recorded_signal[idx][
                :self.live_msrmt_data.record_idx
            ]
        if self.state in [MsrmtState.FINISHING, MsrmtState.FINISHED]:

            return self.live_msrmt_data.recorded_signal[idx]
        return np.zeros(0,np.float32)

    def msrmt_callback(
        self,
        input_data: npt.NDArray[np.floating],
        output_data: npt.NDArray[np.floating],
        frames: int,
        time: Any,
        status: sd.CallbackFlags
    ) -> None:
        """Callback for the measurement

        Callback that is called when another block of measurement data is
        available and the next input data needs to be set.

        Args:
            input_data: A 2D matrix of shape (frames, n_input_channels)
              with recorded data from device
            output_data: A 2D matrix of shape (frames, n_output_channels)
              that needs to be filled with data to record
            frames: Number of frames that need to be read/written
            time: object containing info about timings of AD/DA conversions
            status: Status containing information about the callback

        """
        del time
        del status

        # Set the end of the measurement index for this callback
        start_idx = self.live_msrmt_data.play_idx
        end_idx = start_idx + frames

        rec_start_idx = self.live_msrmt_data.record_idx
        rec_end_idx = rec_start_idx + frames

        self.live_msrmt_data.record_idx += frames
        self.live_msrmt_data.play_idx += frames

        # Set output data to device buffer depending on measurement state
        match self.state:
            case MsrmtState.SYNCING:
                data = self.live_msrmt_data.sync_output[start_idx:end_idx]

                # If sync signal has been finished, pad with zeros
                # NB: Currently not used, because sync samples are a
                # multiple of device block size
                # if len(data) < frames:
                #     data = np.pad(data, (0, frames-len(data)))
                data_stereo = np.zeros(
                    (frames, self.hardware_data.get_stream_output_channels())
                )
                # Add data to sync output channel
                data_stereo[:, DeviceConfig.sync_channels[0]] = data
                output_data[:] = data_stereo

            case MsrmtState.RECORDING | MsrmtState.STARTING:
                # Convert each signal to output data
                chunks = np.zeros(
                    (frames, len(self.live_msrmt_data.output_signals)),
                    dtype=np.float32
                )

                for i, signal_i in enumerate(
                    self.live_msrmt_data.output_signals
                ):
                    is_finished = signal_i.get_data(
                        start_idx,
                        end_idx,
                        signal_buffer=chunks[:,i]
                    )

                    if is_finished:
                        self.set_state(MsrmtState.END_RECORDING)
                output_data[:] = chunks

            case _:
                # play mute signal during setup and when finished
                output_data[:] = 0

        # Handle the input data of the device buffer and the
        # control parameters of the measurement flow


        match self.state:
            case MsrmtState.SETUP:
                # clear buffer of audio device during setup
                # by playing and recording mute signal
                self.i_init_runs += 1
                if self.i_init_runs == NUM_INIT_RUNS:
                    frames = 0
                    self.live_msrmt_data.play_idx = frames
                    self.live_msrmt_data.record_idx = 0

                    self.set_state(MsrmtState.SYNCING)
            case MsrmtState.SYNCING:
                if end_idx < len(self.live_msrmt_data.sync_recorded):
                    self.live_msrmt_data.sync_recorded[
                        rec_start_idx:rec_end_idx
                    ] = input_data[:frames, DeviceConfig.sync_channels[1]]

                else:
                    # End of sync recording reached
                    # Store remaining frames, compute latency,
                    # and transition to next state
                    num_remaining_frames = (
                        len(self.live_msrmt_data.sync_recorded) - rec_start_idx
                    )
                    rec_end_idx = rec_start_idx + num_remaining_frames
                    self.live_msrmt_data.sync_recorded[rec_start_idx:] = (
                        input_data[:num_remaining_frames, DeviceConfig.sync_channels[1]]
                    )
                    self.set_state(MsrmtState.COMPUTING)
                self.live_msrmt_data.record_idx = rec_end_idx

            case MsrmtState.COMPUTING:
                self.compute_latency()
                if self.live_msrmt_data.latency_samples <= 0:
                    self.set_state(MsrmtState.CANCELED)
                    self.logger.error(
                        'Invalid latency - measurement canceled.'
                    )
                else:
                    self.set_state(MsrmtState.STARTING)
                    frames = 0
                    self.live_msrmt_data.play_idx = frames
                    self.live_msrmt_data.record_idx = 0

            case MsrmtState.STARTING:
                # latency computation successful
                # ignore data before latency has been compensated

                msrmt_start_idx = self.live_msrmt_data.latency_samples

                if USE_EARLY_LATENCY_CORRECTION:
                    # try to detect an early measurement start
                    # (see `USE_EARLY_LATENCY_CORRECTION` for
                    # explanation)
                    max_monitor = np.max(input_data[:, DeviceConfig.sync_channels[1]])
                    if self.monitoring_amp.size:
                        # compare current maximum to average in previous blocks
                        signal_thresh = 1.5 * np.mean(self.monitoring_amp)
                        is_signal = max_monitor > signal_thresh
                        self.logger.debug(
                            'Monitoring amp.: %.5f | threshold: %.5f',
                            max_monitor,
                            signal_thresh
                        )
                    else:
                        is_signal = False
                    self.monitoring_amp = np.append(
                        self.monitoring_amp, max_monitor
                    )

                    if is_signal and rec_end_idx < msrmt_start_idx:
                        msrmt_start_idx -= self.recording_data.block_size
                        self.logger.warning(
                            'Early signal detected! New start index: %d.'
                            f': {msrmt_start_idx}'
                        )

                if rec_start_idx <= msrmt_start_idx <= rec_end_idx:
                    # input frames contains beginning of measurement data
                    self.logger.debug(
                        'Start measurement at sample index %d.',
                        rec_start_idx
                    )
                    num_remaining_frames = rec_end_idx - msrmt_start_idx
                    counter = 0
                    for i in range(self.hardware_data.n_in_channels):
                        if i in self.hardware_data.get_unique_input_channels():
                            self.live_msrmt_data.recorded_signal[counter][
                                :num_remaining_frames
                            ] = input_data[frames - num_remaining_frames:, i]
                            counter += 1

                    self.live_msrmt_data.record_idx = num_remaining_frames
                    self.msrmt_events.enable_plot.set()
                    self.set_state(MsrmtState.RECORDING)

            case MsrmtState.RECORDING | MsrmtState.END_RECORDING:
                # In the recording, write the recorded data into the array;
                # synchronized recording with compensated latency;
                # this is the standard acquisition mode.
                if rec_end_idx >= self.recording_data.msrmt_samples:
                    # we have reached the end of the measurement
                    num_remaining_frames = (
                        self.recording_data.msrmt_samples - rec_start_idx
                    )
                    counter = 0
                    for i in range(self.hardware_data.n_in_channels):
                        if i in self.hardware_data.get_unique_input_channels():
                            self.live_msrmt_data.recorded_signal[counter][
                                rec_start_idx:rec_start_idx+num_remaining_frames
                            ] = input_data[:num_remaining_frames, i]
                            counter += 1
                    self.set_state(MsrmtState.FINISHING)
                else:
                    counter = 0
                    for i in range(self.hardware_data.n_in_channels):
                        if i in self.hardware_data.get_unique_input_channels():
                            self.live_msrmt_data.recorded_signal[counter][
                                rec_start_idx:rec_end_idx] = input_data[:frames, i]
                            counter += 1

                self.live_msrmt_data.record_idx = rec_end_idx
            case MsrmtState.FINISHED:
                raise sd.CallbackStop

        with self.record_idx_share.get_lock():
            self.record_idx_share.value = self.live_msrmt_data.record_idx

    def _finalize_measurement(self) -> None:
        """Finalize measurement after recording is complete."""
        self.logger.info("Finalizing measurement.")
        self.set_state(MsrmtState.FINISHED)

    def run_stream(self):
        """Run the audio stream."""
        try:
            with sd.Stream(
                samplerate=self.recording_data.fs,
                blocksize=self.recording_data.block_size,
                channels=(
                    self.hardware_data.get_stream_input_channels(),
                    self.hardware_data.get_stream_output_channels()
                ),
                dtype='float32',
                callback=self.msrmt_callback,
                latency=self.live_msrmt_data.latency_type,
                device=(
                    self.hardware_data.input_device,
                    self.hardware_data.output_device
                )
            ):
                self.logger.info('Beginning to stream.')

                # Keep stream alive until measurement ends

                while True:
                    if self.state == MsrmtState.FINISHING:
                        self._finalize_measurement()
                        break

                    if self.state in (
                        MsrmtState.FINISHED,
                        MsrmtState.CANCELED,
                    ):
                        break
                    if self.msrmt_events.cancel_msrmt.is_set():
                        self.set_state(MsrmtState.CANCELED)
                        break

                    sleep(0.01)


        finally:
            self.logger.info('Closing stream.')

    def start_plot_process(self) -> None:
        """Start plotting in a separate process"""
        self.plot_process = LivePlotProcess(
            self.shm,
            self.record_idx_share,
            self.recording_data.msrmt_samples,
            self.recording_data.fs,
            self.msrmt_events,
            self.block_duration
        )

    def run_msrmt(self) -> None:
        """Setup, run the measurement and close shared memories afterwards"""

        self.start_plot_process()

        self.run_stream()

        if hasattr(self, 'plot_process'):
            self.plot_process.stop()

        if hasattr(
            self, 'shm'
        ):
            for i, shm_i in enumerate(self.shm):
                recorded_copy = self.live_msrmt_data.recorded_signal[i][
                    :self.live_msrmt_data.record_idx
                ].copy()
                shm_i.close()
                shm_i.unlink()
                self.live_msrmt_data.recorded_signal[i] = recorded_copy
