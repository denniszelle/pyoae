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

    max_digital_output: ClassVar[float] = 0.5
    """Limit for the digital output amplitude re full scale.

    If the peak amplitude of any output exceeds this limit,
    a warning will be displayed and the output limited to
    the maximum digital output amplitude specified by this
    value.
    """

    update_interval: ClassVar[float] = 100.
    """Interval in milliseconds for updating the plots."""

    enable_output_phase_calib: bool = False
    """Flag to enable or disable the output phase calibration."""

    sync_channels: ClassVar[list[int]] = [0, 1]
    """Sync channel of format [output_channel, input_channel]"""

    output_input_mapping: ClassVar[list[list[int]]] = [[0, 0], [1, 0]]
    """Mapping of output channels to input channels"""

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
            Device configuration:
            Input: {self.input_device}
            Output: {self.output_device}
            Sample rate: {self.sample_rate} Hz
            Device buffer size: {self.device_buffer_size}
            Use ramp: {self.use_ramp}
            Ramp duration: {self.ramp_duration} ms
            Update interval: {self.update_interval} ms
        '''
