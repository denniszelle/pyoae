"""Functions and classes for a synchronized measurement."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Literal, Final

import sounddevice as sd
import numpy as np
import numpy.typing as npt
import scipy.signal
from scipy.signal import windows


SYNC_WIDTH: Final[float] = 5.
"""Width of the sync pulse in milliseconds."""

SYNC_DURATION: Final[float] = 1.0
"""Duration of the sync signal (with mute) in seconds."""

SYNC_START: Final[float] = 100.
"""Start time in milliseconds of the sync pulse in the sync signal."""

SYNC_CROSS: Final[int] = 50
"""Number of zero crossings in sync signal."""

SYNC_AMPLITUDE: Final[float] = 0.1
"""Amplitude of the sync signal in digital full scale."""

SYNC_FREQUENCY: Final[float] = 4000
"""Carrier frequency of the sync pulse in Hz."""

SYNC_OUTPUT_CHANNEL: Final[int] = 1
"""Output channel of the sync signal."""

SYNC_INPUT_CHANNEL: Final[int] = 1
"""Input channel to record the sync signal."""

NUM_INIT_RUNS: Final[int] = 3
"""Number of initial runs to clear device buffers.

A measurement callback by the sound device is considered
a run.
"""


def generate_sync(fs: float) -> npt.NDArray[np.float32]:
    """Generates and returns the sync signal.

    Args:
        fs: Sampling frequency in Hz

    Returns:
        sync_pulse: 1D array of sync pulse signal
    """
    k = np.ceil(fs / (4.0 * SYNC_FREQUENCY))
    samples_per_sync_period = 4.0 * k
    f_sync = fs / samples_per_sync_period
    p = np.ceil((SYNC_CROSS + 1) / 2)
    t_off = (p / f_sync) - 1/fs

    num_samples = int(t_off * fs)
    t = np.arange(num_samples) / fs
    y = np.sin(2 * np.pi * f_sync * t)
    w = windows.tukey(num_samples)
    sync_pulse = SYNC_AMPLITUDE * w * y

    return sync_pulse


class Signal(ABC):
    """Abstract base class for output signals."""

    @abstractmethod
    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        is_stop: bool = False
    ) -> npt.NDArray[np.float32]:
        """Returns data of the signal."""


@dataclass
class SimpleSignal(Signal):
    """Simple signal for output."""

    signal_data: npt.NDArray[np.float32]
    """1D NumPy array with the output signal."""

    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        is_stop: bool = False
    ) -> npt.NDArray[np.float32]:
        """Returns data of the signal"""
        if end_idx>len(self.signal_data):
            raise IndexError('Signal out of bounds.')
        return self.signal_data[start_idx:end_idx]


@dataclass
class PeriodicSignal(Signal):
    """Periodic output signal.

    Represents a harmonic signal and its periodic
    continuation.
    """

    signal_data: npt.NDArray[np.float32]
    """1D NumPy array with the harmonic output signal."""

    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        is_stop: bool = False
    ) -> npt.NDArray[np.float32]:
        """Returns frames of a periodic signal.

        Returns the data with continuation.
        """
        num_block_samples = len(self.signal_data)
        length = end_idx - start_idx
        start_mod = start_idx % num_block_samples
        end_mod = (start_mod + length) % num_block_samples

        if start_mod + length <= num_block_samples:
            # Single slice, no wraparound
            return self.signal_data[start_mod:start_mod + length]
        # Wraps around: concatenate tail and head
        tail = self.signal_data[start_mod:]
        if is_stop:
            return tail
        head = self.signal_data[:end_mod]
        return np.concatenate((tail, head))


@dataclass
class PeriodicRampSignal(PeriodicSignal):
    """Periodic signal with fade-in and fade-out ramps."""

    n_recording_samples: int
    """Number of total samples of the recording."""

    ramp: npt.NDArray[np.float32]
    """1D NumPy array with the envelope of the ramp."""

    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        is_stop: bool = False
    ) -> npt.NDArray[np.float32]:
        """Returns data of the periodic continuation of the signal.

        At the beginning and the end of the recording, the output
        is multiplied with the ramp envelope to fade-in and
        fade-out the signal, respectively.
        """

        if start_idx > self.n_recording_samples:
            return np.zeros(end_idx-start_idx, dtype=np.float32)

        data = super().get_data(start_idx, end_idx)

        # Apply fade-in if in the fade-in region
        if start_idx < len(self.ramp):
            fade_start = max(0, start_idx)
            fade_end = min(end_idx, len(self.ramp))
            ramp_start = fade_start
            ramp_end = fade_end
            data[0:fade_end - start_idx] *= self.ramp[ramp_start:ramp_end]

        # Apply fade-out if in the fade-out region (end of the array)
        fade_out_start = self.n_recording_samples - len(self.ramp)
        if end_idx > fade_out_start:
            fade_start = max(start_idx, fade_out_start)
            fade_end = min(end_idx, self.n_recording_samples)

            ramp_start = fade_start - fade_out_start
            ramp_end = fade_end - fade_out_start
            seg_start = fade_start - start_idx
            seg_end = fade_end - start_idx

            # Ensure that we actually have something to apply
            if ramp_end > ramp_start and seg_end > seg_start:
                data[seg_start:seg_end] *= self.ramp[::-1][ramp_start:ramp_end]

        if end_idx > self.n_recording_samples:
            data[self.n_recording_samples-end_idx:] = 0

        return data


class MsrmtState(Enum):
    """State of the measurement."""

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

    MEASURING = auto()
    """In main measurement."""

    END_MEASURING = auto()

    FINISHING = auto()
    """Completing main measurement."""

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
class LiveMsrmtData:
    """Variable storage for the live measurement."""

    play_idx: int
    """Raw measurement index."""

    record_idx: int

    latency_samples: int
    """DAQ latency of the measurement device in samples.

    The latency of the data acquisition is determined by presenting
    a tone burst on SYNC_OUTPUT_CHANNEL and recording it via an
    electric connection (short circuit) on SYNC_IN_CHANNEL.

    By presenting a mute signal of length latency_samples,
    output and input in the main measurement are synchronized.
    """

    output_signals: list[Signal]
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


class SyncMsrmt:
    """Class to perform a synchronized OAE measurement."""

    state: MsrmtState
    """The current state of the measurement.

    Used to control the measurement sequence.
    """

    recording_data: RecordingData
    """Object containing information about the recording."""

    hardware_data: HardwareData
    """Object containing information about the hardware."""

    live_msrmt_data: LiveMsrmtData
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

    def __init__(
        self,
        recording_data: RecordingData,
        hardware_data: HardwareData,
        output_signals: list[npt.NDArray[np.float32] | Signal],
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

        sync_output = np.zeros(
            int(self.recording_data.fs * SYNC_DURATION),
            dtype=np.float32
        )
        sync_recorded = np.zeros(
            int(self.recording_data.fs * SYNC_DURATION),
            dtype=np.float32
        )
        sync_pulse = generate_sync(self.recording_data.fs)
        sync_output[:len(sync_pulse)] = sync_pulse
        recorded_signal = np.zeros(
            self.recording_data.msrmt_samples,
            dtype=np.float32
        )
        self.live_msrmt_data = LiveMsrmtData(
            play_idx=0,
            record_idx=0,
            latency_samples=-1,
            output_signals=[],
            sync_output=sync_output,
            sync_recorded=sync_recorded,
            latency_type=latency_type,
            recorded_signal=recorded_signal
        )
        self.i_init_runs = 0
        self.set_output_signals(output_signals)

    def set_state(self, state: MsrmtState) -> None:
        """Sets the currently active state of this measurement."""
        self.state = state
        print(f"Measurement state set to {self.state}.")

    def set_output_signals(
        self,
        signals: list[npt.NDArray[np.float32] | Signal]
    ) -> None:
        """Sets the output signal for the main measurement."""
        for signal_i in signals:
            if isinstance(signal_i, Signal):
                self.live_msrmt_data.output_signals.append(signal_i)
            else:
                if len(signal_i) != self.recording_data.msrmt_samples:
                    raise ValueError(
                        'Input signal duration must match measurement duration'
                    )
                self.live_msrmt_data.output_signals.append(
                    SimpleSignal(signal_i)
                )

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
        correlation = scipy.signal.correlate(
            self.live_msrmt_data.sync_recorded,
            self.live_msrmt_data.sync_output
        )
        num_acq_samples = len(self.live_msrmt_data.sync_output)
        lag = np.argmax(correlation) - num_acq_samples + 1
        latency_time = (lag/self.recording_data.fs) * 1E3

        print(f'Measured latency: {latency_time:.4f} ms ({lag} samples)')
        if not isinstance(lag, np.integer):
            print(
                'Warning: Cannot obtain latency.  '
                f'Invalid type {type(lag)}.'
            )
            self.live_msrmt_data.latency_samples = 0
            return

        lag = int(lag)
        if lag < 0:
            print('Warning: Cannot obtain latency. Negative value detected.')
            print('Please check cable connections.')
        self.live_msrmt_data.latency_samples = max(int(lag), 0)

    def get_recorded_signal(self) -> npt.NDArray[np.float32]:
        """Returns the recorded signal.

        Returns:
            Recorded signal with the length of already recorded data

        """
        if self.state in [MsrmtState.MEASURING, MsrmtState.END_MEASURING] :
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
                if len(data) < frames:
                    data = np.pad(data, (0, frames-len(data)))

                data_stereo = np.zeros(
                    (frames, self.hardware_data.n_out_channels)
                )
                # Add data to sync output channel
                data_stereo[:,SYNC_OUTPUT_CHANNEL-1] = data
                output_data[:] = data_stereo

            case MsrmtState.MEASURING | MsrmtState.STARTING:
                # Convert each signal to output data
                chunks = []
                i_current_block = (start_idx // self.recording_data.num_block_samples) + 1
                num_max_avg = int(self.recording_data.msrmt_samples / self.recording_data.num_block_samples)
                is_stop = i_current_block == num_max_avg

                for signal_i in self.live_msrmt_data.output_signals:
                    chunk_i = signal_i.get_data(
                        start_idx,
                        end_idx,
                        is_stop=is_stop
                    )

                    if len(chunk_i) < frames:
                        chunk_i = np.pad(chunk_i, (0, frames-len(chunk_i)))
                        self.set_state(MsrmtState.END_MEASURING)
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
                if self.i_init_runs >= NUM_INIT_RUNS:
                    self.live_msrmt_data.play_idx = 0
                    frames = 0
                    self.live_msrmt_data.record_idx = 0
                    self.set_state(MsrmtState.SYNCING)
                else:
                    self.live_msrmt_data.record_idx += frames
            case MsrmtState.SYNCING:
                if end_idx < len(self.live_msrmt_data.sync_recorded):
                    self.live_msrmt_data.sync_recorded[start_idx:end_idx] = (
                        input_data[:frames, SYNC_INPUT_CHANNEL-1]
                    )
                else:
                    # End of sync recording reached
                    # Store remaining frames, compute latency,
                    # and transition to next state

                    num_remaining_frames = (
                        len(self.live_msrmt_data.sync_recorded) - start_idx
                    )

                    self.live_msrmt_data.sync_recorded[start_idx:] = (
                        input_data[:num_remaining_frames, SYNC_INPUT_CHANNEL-1]
                    )
                    self.set_state(MsrmtState.COMPUTING)

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
                if rec_start_idx <= msrmt_start_idx <= rec_end_idx:
                    # input frames contains beginning of measurement data
                    num_remaining_frames = rec_end_idx - msrmt_start_idx
                    self.live_msrmt_data.recorded_signal[:num_remaining_frames] = (
                        input_data[frames - num_remaining_frames:, 0]
                    )
                    # set
                    self.live_msrmt_data.record_idx = num_remaining_frames
                    self.set_state(MsrmtState.MEASURING)
                    # self.live_msrmt_data.play_idx = 0
                    # frames = 0
                else:
                    self.live_msrmt_data.record_idx += frames

            case MsrmtState.MEASURING | MsrmtState.END_MEASURING:
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
