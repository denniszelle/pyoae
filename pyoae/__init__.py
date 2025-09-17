"""PyOAE - A python package to record Otoacoustic Emissions.

This package comprises the following submodules that provide the
backend for synchronous acquisition of otoacoustic emissions
using `python-sounddevice` to access an audio device.

Submodules:

- `pyoae.calib`: Calibration module (currently a dummy module with no function)
- `pyoae.cdpoae`: Module to record continuous DPOAE
- `pyoae.soae`: Module to record SOAE (no stimulus)
- `pyoae.sync`: Module with basic functions and classes for synchronized
     audio output/input

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
