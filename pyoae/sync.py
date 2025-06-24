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


SYNC_WIDTH: Final[float] = 5.
"""Width of the sync pulse in milliseconds."""

SYNC_DURATION: Final[float] = 1.0
"""Duration of the sync signal (with mute) in seconds."""

SYNC_START: Final[float] = 100.
"""Start time in milliseconds of the sync pulse in the sync signal."""

SYNC_AMPLITUDE: Final[float] = 0.1
"""Amplitude of the sync signal in digital full scale."""

SYNC_FREQUENCY: Final[float] = 4000
"""Carrier frequency of the sync pulse in Hz."""

SYNC_OUTPUT_CHANNEL: Final[int] = 1
"""Output channel of the sync signal."""

SYNC_INPUT_CHANNEL: Final[int] = 2
"""Input channel to record the sync signal."""


def generate_sync(fs: float) -> npt.NDArray[np.float32]:
    """Generates and returns the sync signal.

    Args:
        fs: Sampling frequency in Hz

    Returns:
        pulse: 1D array of sync pulse signal
    """
    n_sync_width = int(np.floor(1E-3*SYNC_WIDTH*fs))
    n_sync_start = int(np.floor(1E-3*SYNC_START*fs))
    t_carrier = np.arange(n_sync_width, dtype=np.float32)/fs
    carrier_sine = np.sin(2*np.pi*SYNC_FREQUENCY*t_carrier)
    pulse = np.zeros(int(fs * SYNC_DURATION), dtype=np.float32)
    pulse[n_sync_start:n_sync_start + n_sync_width] = (
        SYNC_AMPLITUDE*np.hanning(n_sync_width)
    )
    pulse[n_sync_start : n_sync_start+n_sync_width] *= carrier_sine
    return pulse


class Signal(ABC):
    """Abstract base class for output signals."""

    @abstractmethod
    def get_data(
        self,
        start_idx: int,
        end_idx: int
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
        end_idx: int
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
        end_idx: int
    ) -> npt.NDArray[np.float32]:
        """Returns data of the periodic continuation of the signal."""
        length = end_idx-start_idx
        period = len(self.signal_data)
        data = np.zeros(length, dtype=np.float32)
        periods = int(np.floor(length/period))
        data[0:periods*period] = np.tile(
            np.roll(self.signal_data, -start_idx%period),
            periods
        )
        missing_samples = length-periods*period
        if missing_samples:
            data[periods*period:] = np.roll(
                self.signal_data, -start_idx%period
            )[:missing_samples]
        return data


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
        end_idx: int
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

    SYNCING = auto()
    """Sync pulse is currently played."""

    MEASURING = auto()
    """In main measurement."""

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

        self.state = MsrmtState.SETUP
        self.recording_data = recording_data
        self.hardware_data = hardware_data
        self.live_msrmt_data = LiveMsrmtData(
            0,
            -1,
            [],
            generate_sync(self.recording_data.fs),
            np.zeros(
                int(self.recording_data.fs * SYNC_DURATION),
                dtype=np.float32
            ),
            latency_type,
            np.zeros(
                (
                    self.recording_data.msrmt_samples
                    + int(self.recording_data.fs*5)
                ),
                dtype=np.float32
            )
        )

        self.set_output_signals(output_signals)

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
        self.state = MsrmtState.SYNCING
        self.run_stream(start_plot, info)

    def compute_latency(self) -> None:
        """Computes the stream latency from sync measurement."""
        correlation = scipy.signal.correlate(
            self.live_msrmt_data.sync_recorded,
            self.live_msrmt_data.sync_output
        )
        lag = (
            np.argmax(correlation)
            - int(SYNC_DURATION * self.recording_data.fs)
            + 1
        )
        latency_time = lag/self.recording_data.fs

        print(f'Measured latency: {latency_time:.4f} seconds ({lag} samples)')
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
            print('Warning: Cannot obtain latency. Negative value detected.')
        self.live_msrmt_data.latency_samples = max(int(lag), 0)

    def get_recorded_signal(self) -> npt.NDArray[np.float32]:
        """Returns the recorded signal.

        Returns:
            Recorded signal with the length of already recorded data

        """
        if (self.state == MsrmtState.MEASURING
            and self.live_msrmt_data.latency_samples >= 0):
            if (
                self.live_msrmt_data.play_idx
                > self.live_msrmt_data.latency_samples
            ):
                return self.live_msrmt_data.recorded_signal[
                    :(self.live_msrmt_data.play_idx
                    -self.live_msrmt_data.latency_samples)
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

        # Choose actions depending on the state of the measurement
        match self.state:
            case MsrmtState.SYNCING:
                data = self.live_msrmt_data.sync_output[
                    start_idx:end_idx
                ]
                # If sync signal has been finished, pad with zeros and
                # transition to next state
                if len(data) < frames:
                    data = np.pad(data, (0, frames-len(data)))
                    self.state = MsrmtState.MEASURING
                    # Save sync recording with data from sync input channel
                    self.live_msrmt_data.sync_recorded[
                        start_idx:end_idx
                    ] = input_data[
                        :len(self.live_msrmt_data.sync_recorded[
                            start_idx:
                        ]),
                        SYNC_INPUT_CHANNEL-1
                    ]
                else:
                    self.live_msrmt_data.sync_recorded[
                        start_idx:end_idx
                    ] = input_data[
                        :len(data),
                        SYNC_INPUT_CHANNEL-1
                    ]
                data_stereo = np.zeros(
                    (frames, self.hardware_data.n_out_channels)
                )
                # Add data to sync output channel
                data_stereo[:,SYNC_OUTPUT_CHANNEL-1] = data
                output_data[:] = data_stereo

            case MsrmtState.MEASURING:
                # If latency has not yet been computed, compute latency.
                if self.live_msrmt_data.latency_samples < 0:
                    self.compute_latency()
                    self.live_msrmt_data.play_idx = 0
                    start_idx = 0
                    end_idx = start_idx + frames

                # Convert each signal to output data
                chunks = []
                for signal_i in self.live_msrmt_data.output_signals:
                    chunk_i = signal_i.get_data(
                        start_idx, end_idx
                    )
                    if len(chunk_i) < frames:
                        chunk_i = np.pad(chunk_i, (0, frames-len(chunk_i)))
                    chunks.append(chunk_i)
                stereo_chunk = np.column_stack(chunks)
                output_data[:] = stereo_chunk.astype(np.float32)

                # In the recording, write the recorded data into the array
                if ((end_idx + self.live_msrmt_data.latency_samples)
                    <= len(self.live_msrmt_data.recorded_signal)):

                    # If play_idx is after signal start, save everything
                    if start_idx-self.live_msrmt_data.latency_samples > 0:
                        self.live_msrmt_data.recorded_signal[
                            (start_idx - self.live_msrmt_data.latency_samples):
                            (end_idx-self.live_msrmt_data.latency_samples)
                        ] = input_data[:frames, 0]
                    # If play_idx is before signal start, but end of buffer is
                    # after signal start, save a subset of recorded data
                    elif (end_idx-self.live_msrmt_data.latency_samples) > 0:
                        sig_samples = end_idx - self.live_msrmt_data.latency_samples
                        self.live_msrmt_data.recorded_signal[
                            0:sig_samples
                        ] = input_data[
                            frames-sig_samples:frames,
                            0
                        ]
                # If play_idx is after measurement end, finish measurement and
                # crop recorded signal to measurement length
                if ((start_idx - self.live_msrmt_data.latency_samples)
                    > self.recording_data.msrmt_samples):
                    self.state = MsrmtState.FINISHING
                    self.live_msrmt_data.recorded_signal = (
                        self.live_msrmt_data.recorded_signal[
                            :self.recording_data.msrmt_samples]
                    )

            case MsrmtState.FINISHING:
                # Play mute if measurement is finishing
                output_data[:] = 0

            case MsrmtState.FINISHED:
                # Play mute if measurement is finished
                output_data[:] = 0

            case MsrmtState.CANCELED:
                # Play mute if measurement has been canceled
                output_data[:] = 0

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
