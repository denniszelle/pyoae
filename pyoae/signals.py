"""Classes to store and manage playback signals."""

import numpy as np
import numpy.typing as npt


class Signal:
    """Simple base class to store and handle output signals."""

    num_signal_samples: int
    """Number of samples of the provided signal template.

    This is the length of signal data.
    """

    signal_data: npt.NDArray[np.float32]
    """1D NumPy array with the output signal."""

    def __init__(
        self,
        signal_data: npt.NDArray[np.float32]
    ) -> None:
        self.signal_data = signal_data
        self.num_signal_samples = len(signal_data)

    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        signal_buffer: npt.NDArray[np.float32]
    ) -> bool:
        """Writes data into the signal buffer and returns output finish flag"""
        if end_idx < self.num_signal_samples:
            signal_buffer[:] = self.signal_data[start_idx:end_idx]
            return False
        else:
            end_len = self.num_signal_samples-start_idx
            signal_buffer[:end_len] = self.signal_data[start_idx:]
            return True


class PeriodicSignal(Signal):
    """Periodic output signal.

    Represents the periodic continuation of a signal.
    """

    num_total_samples: int
    """Number of total output samples.

    Note:
        Output samples can be reached by signal repetition.
    """

    def __init__(
        self,
        signal_data: npt.NDArray[np.float32],
        num_total_samples: int
    ) -> None:
        super().__init__(signal_data)
        self.num_total_samples = num_total_samples

    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        signal_buffer: npt.NDArray[np.float32]
    ) -> bool:
        """Writes data into the data buffer and returns output finish flag

        Returns the data with continuation.

        Note:
            This overwrites `get_data` from the base class.
        """
        length = end_idx - start_idx
        start_mod = start_idx % self.num_signal_samples
        end_mod = (start_mod + length) % self.num_signal_samples

        if start_mod + length <= self.num_signal_samples:
            # Single slice, no wraparound
            signal_buffer[:] = self.signal_data[start_mod:start_mod + length]
            return False
        # Wraps around: concatenate tail and head
        tail = self.signal_data[start_mod:]
        if end_idx >= (self.num_total_samples - 1):
            # end of requested playback reached
            signal_buffer[:len(tail)] = tail
            return True
        head = self.signal_data[:end_mod]
        signal_buffer[:] = np.concatenate((tail, head))
        return False


class PeriodicRampSignal(PeriodicSignal):
    """Periodic signal with fade-in and fade-out ramps."""

    ramp: npt.NDArray[np.float32]
    """1D NumPy array with the envelope of the ramp."""

    idx_fade_out: int
    """Index at which the fade-out ramp starts."""

    def __init__(
        self,
        signal_data: npt.NDArray[np.float32],
        num_total_samples: int,
        ramp: npt.NDArray[np.float32]
    ) -> None:
        super().__init__(signal_data, num_total_samples)
        self.ramp = ramp
        self.idx_fade_out = self.num_total_samples - len(ramp)

    def get_data(
        self,
        start_idx: int,
        end_idx: int,
        signal_buffer: npt.NDArray[np.float32]
    ) -> bool:
        """Applies fade-in or fade-out and writes data into signal buffer"""
        is_finished = super().get_data(start_idx, end_idx, signal_buffer)

        # Apply fade-in
        if start_idx < len(self.ramp):
            fade_len = min(end_idx, len(self.ramp)) - start_idx
            signal_buffer[:fade_len] *= self.ramp[
                start_idx:start_idx + fade_len
            ]

        # Apply fade-out
        if end_idx > self.idx_fade_out:
            fade_start = max(start_idx, self.idx_fade_out)
            k = fade_start - self.idx_fade_out
            ramp_len = end_idx - fade_start
            available_len = len(self.ramp) - k
            # the effective ramp length is applied if end
            # of output data was reached; i.e., the data
            # was trimmed in super().get_data(...)
            effective_len = min(ramp_len, available_len)
            offset = end_idx-self.num_total_samples

            buffer_slice = signal_buffer[-offset - effective_len : -offset]
            ramp_slice = self.ramp[::-1][k : k + effective_len]
            buffer_slice *= ramp_slice

        return is_finished
