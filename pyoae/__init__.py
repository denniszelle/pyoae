"""PyOAE - A python package to record Otoacoustic Emissions.

This package comprises the following submodules that provide the
backend for synchronous acquisition of otoacoustic emissions
using `python-sounddevice` to access an audio device.

Version: 0.3.0 (January 2026) - www.earlab.de

Submodules:

- `pyoae.abs_calib`: Module to perform absolute calibration of input sensitivity.
- `pyoae.calib`: Module providing functionality for managing calibration data.
- `pyoae.calibrator`: Module to calibrate output channels.
- `pyoae.cdpoae`: Module to record continuous DPOAEs.
- `pyoae.converter`: Module providing basic conversion functions.
- `pyoae.files`: Module for file handling.
- `pyoae.generator`: Module with functions and classes to create output stimuli.
- `pyoae.helpers`: Module with miscellaneous helper functions.
- `pyoae.msrmt_context`: Module with dataclasses storing measurement context.
- `pyoae.pdpoae`: Module to record pulsed DPOAEs.
- `pyoae.plotting`: Module that includes live measurement plot functionality
- `pyoae.protocols`: Module handling measurement and configuration protocols.
- `pyoae.pyoae_logger`: Module that sets up the logger
- `pyoae.signals`: Module managing playback signals.
- `pyoae.soae`: Module to record SOAE (without any stimulus).
- `pyoae.sync`: Module providing synchronized audio output/input.

- `pyoae.device.device_config`: Storage module for device configuration
- `pyoae.device.display_devices`: Function to show detected audio interfaces

- `pyoae.dsp.containers`: Classes to define measurement saving structure
- `pyoae.dsp.opt_avg`: Optimized averaging functionality
- `pyoae.dsp.processing`: Measurement data processing functionality
"""

from __future__ import annotations
import logging

# Prevent "No handler could be found" warnings for library users.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_PKG = __name__

def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger: 'pyoae' or 'pyoae.<name>'."""
    return logging.getLogger(f"{_PKG}.{name}" if name else _PKG)

def set_default_level(level: int) -> None:
    """Lightweight helper for apps/tests.

    Note:
        Libraries should not call basicConfig.
    """
    logging.getLogger(_PKG).setLevel(level)
