"""Functions and classes for a synchronized measurement."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Generic, Literal, Final, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.signal
import sounddevice as sd

from pyoae import generator
from pyoae.signals import Signal

SYNC_DURATION: Final[float] = 0.5
"""Duration of the sync signal (with mute) in seconds.

Will be adjusted to a correspond to a number of samples
that is a multiple of the device's buffer size.
"""

SYNC_OUTPUT_CHANNEL: Final[int] = 0
"""Output channel of the sync signal."""

SYNC_INPUT_CHANNEL: Final[int] = 1
"""Input channel to record the sync signal."""

NUM_INIT_RUNS: Final[int] = 3
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
    a tone burst on SYNC_OUTPUT_CHANNEL and recording it via an
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

    recorded_signal: npt.NDArray[np.float32]
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

    i_init_runs: int
    """Counter of initial measurement callbacks with mute signal.

    In order to clear the input and output buffers of the audio
    interface, during setup a number of measurement callbacks are
    "ignored" by playing a mute signal and neglecting the recorded
    data.

    When this counter reaches the predefined number of initial runs,
    set by `NUM_INIT_RUNS` the measurement state switches to SYNCING.
    """

    monitoring_amp: npt.NDArray[np.float32]
    """Signal amplitude of monitoring channel during playback start."""

    def __init__(
        self,
        recording_data: RecordingData,
        hardware_data: HardwareData,
        output_signals: list[SignalT],
        latency_type: Literal['low', 'high']='high'
    ) -> None:
        """Initializes the object.

        Args:
            recording_data: Provides meta data for the recording.
            hardware_data: Provides meta data for the used hardware.
        	output_signals: List of output signal objects or 1D output
              signal arrays
            latency_type: Latency type used('low' or 'high')
        """
        self.set_state(MsrmtState.SETUP)
        self.recording_data = recording_data
        self.hardware_data = hardware_data

        # compute duration of synchronization in multiples of block size
        k = int(self.recording_data.fs * SYNC_DURATION) // recording_data.block_size
        num_sync_samples = recording_data.block_size * k
        sync_duration = (num_sync_samples / recording_data.fs) * 1E3
        print(
            'Duration of sync-signal with mute: '
            f'{num_sync_samples} ({sync_duration:.2f} ms)'
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
        recorded_signal = np.zeros(
            self.recording_data.msrmt_samples,
            dtype=np.float32
        )
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

    def set_state(self, state: MsrmtState) -> None:
        """Sets the currently active state of this measurement."""
        self.state = state
        print(f"Measurement state set to {self.state}.")

    def start_msrmt(self, start_plot: Callable, info: Any) -> None:
        """Starts the measurement

        Args:
            start_plot: Function that is called with this object as argument
              when the stream has been started
            info: Info contained in an object that store information for the
              update routine.
        """
        self.run_stream(start_plot, info)

    def compute_latency(self) -> None:
        """Computes the stream latency from sync measurement."""
        self.live_msrmt_data.sync_recorded /= np.max(
            np.abs(self.live_msrmt_data.sync_recorded)
        )
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

        print(
            f'Measured latency: {latency_time:.4f} ms ({lag} samples) '
            f'({num_acq_samples} samples acquired)'
        )
        if not isinstance(lag, np.integer):
            print(
                'Warning: Cannot obtain latency.  '
                f'Invalid type {type(lag)}.'
            )
            self.live_msrmt_data.latency_samples = 0
            return

        if lag < 0:
            print('Warning: Cannot obtain latency. Negative value detected.')
            print('Please check cable connections.')
        self.live_msrmt_data.latency_samples = max(int(lag), 0)

    def get_recorded_signal(self) -> npt.NDArray[np.float32]:
        """Returns the recorded signal.

        Returns:
            Recorded signal with the length of already recorded data

        """
        if self.state in [MsrmtState.RECORDING, MsrmtState.END_RECORDING] :
            return self.live_msrmt_data.recorded_signal[
                :self.live_msrmt_data.record_idx
            ]
        if self.state in [MsrmtState.FINISHING, MsrmtState.FINISHED]:
            return self.live_msrmt_data.recorded_signal
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
                    (frames, self.hardware_data.n_out_channels)
                )
                # Add data to sync output channel
                data_stereo[:, SYNC_OUTPUT_CHANNEL] = data
                output_data[:] = data_stereo

            case MsrmtState.RECORDING | MsrmtState.STARTING:
                # Convert each signal to output data
                chunks = []

                for signal_i in self.live_msrmt_data.output_signals:
                    chunk_i = signal_i.get_data(
                        start_idx,
                        end_idx
                    )

                    if len(chunk_i) < frames:
                        chunk_i = np.pad(chunk_i, (0, frames-len(chunk_i)))
                        self.set_state(MsrmtState.END_RECORDING)
                    chunks.append(chunk_i)
                stereo_chunk = np.column_stack(chunks)
                output_data[:] = stereo_chunk.astype(np.float32)

            case _:
                # play mute signal during setup and when finished
                output_data[:] = 0

        # Handle the input data of the device buffer and the
        # control parameters of the measurement flow

        rec_start_idx = self.live_msrmt_data.record_idx
        rec_end_idx = rec_start_idx + frames

        match self.state:
            case MsrmtState.SETUP:
                # clear buffer of audio device during setup
                # by playing and recording mute signal
                self.i_init_runs += 1
                if self.i_init_runs == NUM_INIT_RUNS:
                    self.live_msrmt_data.play_idx = 0
                    frames = 0
                    self.live_msrmt_data.record_idx = 0
                    self.set_state(MsrmtState.SYNCING)
                else:
                    self.live_msrmt_data.record_idx += frames
            case MsrmtState.SYNCING:
                if end_idx < len(self.live_msrmt_data.sync_recorded):
                    self.live_msrmt_data.sync_recorded[rec_start_idx:rec_end_idx] = (
                        input_data[:frames, SYNC_INPUT_CHANNEL]
                    )
                else:
                    # End of sync recording reached
                    # Store remaining frames, compute latency,
                    # and transition to next state
                    num_remaining_frames = (
                        len(self.live_msrmt_data.sync_recorded) - rec_start_idx
                    )
                    rec_end_idx = rec_start_idx + num_remaining_frames
                    self.live_msrmt_data.sync_recorded[rec_start_idx:] = (
                        input_data[:num_remaining_frames, SYNC_INPUT_CHANNEL]
                    )
                    self.set_state(MsrmtState.COMPUTING)
                self.live_msrmt_data.record_idx = rec_end_idx

            case MsrmtState.COMPUTING:
                self.compute_latency()
                if self.live_msrmt_data.latency_samples <= 0:
                    self.set_state(MsrmtState.CANCELED)
                    print('Invalid latency - measurement canceled.')
                else:
                    self.set_state(MsrmtState.STARTING)
                    self.live_msrmt_data.play_idx = 0
                    self.live_msrmt_data.record_idx = 0
                    frames = 0

            case MsrmtState.STARTING:
                # latency computation successful
                # ignore data before latency has been compensated

                msrmt_start_idx = self.live_msrmt_data.latency_samples

                if USE_EARLY_LATENCY_CORRECTION:
                    # try to detect an early measurement start
                    # (see `USE_EARLY_LATENCY_CORRECTION` for
                    # explanation)
                    max_monitor = np.max(input_data[:, SYNC_INPUT_CHANNEL])
                    if self.monitoring_amp.size:
                        # compare current maximum to average in previous blocks
                        signal_thresh = 1.5 * np.mean(self.monitoring_amp)
                        is_signal = max_monitor > signal_thresh
                        print(
                            f'Monitoring amp.: {max_monitor}, '
                            f'threshold: {signal_thresh}'
                        )
                    else:
                        is_signal = False
                    self.monitoring_amp = np.append(
                        self.monitoring_amp, max_monitor
                    )

                    if is_signal and rec_end_idx < msrmt_start_idx:
                        msrmt_start_idx -= self.recording_data.block_size
                        print(
                            'Early signal detected: correcting '
                            f'latency to: {msrmt_start_idx}'
                        )

                if rec_start_idx <= msrmt_start_idx <= rec_end_idx:
                    # input frames contains beginning of measurement data
                    print(f'Start compensating at {rec_start_idx}.')
                    num_remaining_frames = rec_end_idx - msrmt_start_idx
                    self.live_msrmt_data.recorded_signal[:num_remaining_frames] = (
                        input_data[frames - num_remaining_frames:, 0]
                    )
                    self.live_msrmt_data.record_idx = num_remaining_frames
                    self.set_state(MsrmtState.RECORDING)
                else:
                    self.live_msrmt_data.record_idx += frames

            case MsrmtState.RECORDING | MsrmtState.END_RECORDING:
                # In the recording, write the recorded data into the array;
                # synchronized recording with compensated latency;
                # this is the standard acquisition mode.
                if rec_end_idx >= self.recording_data.msrmt_samples:
                    # we have reached the end of the measurement
                    num_remaining_frames = self.recording_data.msrmt_samples - rec_start_idx
                    self.live_msrmt_data.recorded_signal[
                        rec_start_idx:rec_start_idx+num_remaining_frames
                    ] = input_data[:num_remaining_frames, 0]
                    self.set_state(MsrmtState.FINISHING)
                else:
                    self.live_msrmt_data.recorded_signal[
                        rec_start_idx:rec_end_idx] = input_data[:frames, 0]

                self.live_msrmt_data.record_idx = rec_end_idx

        # Update play_idx
        self.live_msrmt_data.play_idx += frames

    def run_stream(self, start_plot: Callable, info: Any) -> None:
        """Run the measurement stream."""
        with sd.Stream(
            samplerate=self.recording_data.fs,
            blocksize=self.recording_data.block_size,
            channels=(
                self.hardware_data.n_in_channels,
                self.hardware_data.n_out_channels
            ),
            dtype='float32',
            callback=self.msrmt_callback,
            latency=self.live_msrmt_data.latency_type,
            device = (
                self.hardware_data.input_device,
                self.hardware_data.output_device
            )
        ):
            print('Begin stream')
            start_plot(self, info)
            while self.state not in [
                MsrmtState.FINISHING,
                MsrmtState.FINISHED,
                MsrmtState.CANCELED
            ]:
                print(self.state)
                sd.sleep(1000)
            print('Closing stream')
