"""Functions and classes to generate and manage output signals."""

from typing import overload

import numpy as np
import numpy.typing as npt


@overload
def db_to_lin(level: float) -> float: ...
@overload
def db_to_lin(level: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...

def db_to_lin(
    level: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.floating] | float:
    """Convert logarithmic value from dB to linear.

    This function can be used to scale from dBFS to peak amplitude.
    """
    return 10**(level/20)


@overload
def lin_to_db(amplitude: float) -> float: ...
@overload
def lin_to_db(amplitude: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...

def lin_to_db(
    amplitude: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.floating] | float:
    """Convert linear value from linear to dB.

    This function can be used to scale from peak amplitude to dBFS.
    """
    return 20*np.log10(amplitude)


@overload
def db_spl_to_peak_mupa(level: float) -> float: ...
@overload
def db_spl_to_peak_mupa(
    level: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    ...

def db_spl_to_peak_mupa(
    level: float | npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    """Converts sound pressure level to peak signal amplitude in muPa."""
    return 20 * np.sqrt(2) * 10**(level/20)


@overload
def db_spl_to_rms_mupa(level: float) -> float: ...
@overload
def db_spl_to_rms_mupa(
    level: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]: ...

def db_spl_to_rms_mupa(
    level: float | npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    """Converts sound pressure level to signal RMS in muPa."""
    return 20 * 10**(level/20)


@overload
def peak_mupa_to_db_spl(amplitude: float) -> float: ...
@overload
def peak_mupa_to_db_spl(
    amplitude: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    ...

def peak_mupa_to_db_spl(
    amplitude: float | npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    """Converts sound pressure level to peak signal amplitude in muPa."""
    # return 20 * np.sqrt(2) * 10**(amplitude/20)
    return 20*np.log10(amplitude/20)/np.sqrt(2)


@overload
def rms_mupa_to_db_spl(amplitude: float) -> float: ...
@overload
def rms_mupa_to_db_spl(
    amplitude: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    ...

def rms_mupa_to_db_spl(
    amplitude: float | npt.NDArray[np.floating]
) -> float | npt.NDArray[np.floating]:
    """Converts sound pressure level to peak signal amplitude in muPa."""
    # return 20 * np.sqrt(2) * 10**(amplitude/20)
    return 20*np.log10(amplitude/20)
