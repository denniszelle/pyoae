"""Configuration of acquisition device.

This module defines the settings of the acquisition
device, i.e., the sound card.


Note:
    Sound device IDs or names should be known beforehand and
    can be obtained using the display_devices script.
"""


INPUT_DEVICE: int | str = 'Input'
"""Index or name of the input device."""

OUTPUT_DEVICE: int | str = 'Output'
"""Index or name of the output device."""

FS: float = 96000.
"""Sampling frequency in Hz.

Sampling frequency must match that of the device set
in the system settings (Windows) or MIDI setup (macOS).
"""

DEVICE_BUFFER_SIZE: int = 4096
"""Device buffer size used for a single device callback.

Buffer size might need to be increased with increasing
sampling rate to allow for continuous audio output
and data acquisition without gaps.
"""
