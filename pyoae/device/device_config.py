"""Configuration of acquisition device.

This module defines the settings of the acquisition
device, i.e., the sound card.


Note:
    Sound device IDs or names should be known beforehand and
      can be obtained using the display_devices script.
"""

from typing import ClassVar


class DeviceConfig:
    """Class with global DAQ configuration.

    The class variables distribute settings that
    are set dynamically during program start across all
    modules.
    """

    _locked: ClassVar[bool] = False

    input_device: ClassVar[int | str] = 'Input'
    """Index or name of the input device."""

    output_device: ClassVar[int | str] = 'Output'
    """Index or name of the output device."""

    sample_rate: ClassVar[float] = 48000.0
    """Device sample rate in Hz.

    Sampling frequency must match that of the device set
    in the system settings (Windows) or MIDI setup (macOS).
    """

    device_buffer_size: ClassVar[int] = 4096
    """Device buffer size used for a single device callback.

    Buffer size might need to be increased with increasing
    sampling rate to allow for continuous audio output
    and data acquisition without gaps.
    """

    use_ramp: ClassVar[bool] = True
    """Flag enabling/disabling ramps at the begin/end of playback."""

    ramp_duration: ClassVar[float] = 5.0
    """Ramp duration in ms at begin/end of playback."""

    update_interval: ClassVar[float] = 100.
    """Interval in milliseconds for updating the plots."""

    live_display_duration: ClassVar[float] = 100.
    """Duration that will be displayed in the live-time plot in milliseconds."""

    @classmethod
    def set(cls, key, value) -> None:
        """Set a configuration attribute."""
        if cls._locked:
            raise RuntimeError(
                f'Cannot modify DeviceConfig after locking (tried to set "{key}")'
            )
        if not hasattr(cls, key):
            raise AttributeError(
                f'DeviceConfig has no attribute "{key}"'
            )
        setattr(cls, key, value)

    @classmethod
    def lock(cls) -> None:
        """Lock the device configuration to prevent modification."""
        cls._locked = True

    def __str__(self) -> str:
        """Returns device configuration as text."""
        return f'''
            Device configuration: \n
            Input: {self.input_device} \n
            Output: {self.output_device} \n
            Sample rate: {self.sample_rate} Hz \n
            Device buffer size: {self.device_buffer_size} \n
            Use ramp: {self.use_ramp} \n
            Ramp duration: {self.ramp_duration} ms \n
            Update interval: {self.update_interval} ms \n
            Live display: {self.live_display_duration} ms \n
        '''
