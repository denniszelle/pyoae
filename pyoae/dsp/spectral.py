"""Module with functions for spectral signal analysis."""

import numpy as np
import numpy.typing as npt


def bin_coherence(h: np.ndarray) -> float:
    """Estimates the phase coherence of a frequency bin.

    Args:
        h: histogram of phases in a frequency bin across blocks.

    Returns:
        Coherence value between 0 and 1, where 1 indicates
          perfect phase alignment.
    """
    n_k =len(h)
    return np.sqrt(np.sum(h**2)/n_k)


def bin_phases(
    cplx_spectra: np.ndarray,
    bins: np.ndarray
) -> np.ndarray:
    """Compute phases in spectral bins.

    Args:
        cplx_spectra: complex spectra of shape (num_blocks, num_bins).
        bins: indices of bins where phases shall be calculated.

    Returns:
        Array of shape (num_blocks, num_bins) containing the
          phases of the complex spectra in the specified bins.
    """
    return np.angle(cplx_spectra[:, bins])


def coherence_spectrum(
    phases: np.ndarray
) -> np.ndarray:
    """Computes the coherence for spectral bins.

    Args:
        phases: Array of shape (num_blocks, num_bins)
          containing the phases of the complex spectra
          in the specified bins.

    Returns:
        Array with spectral coherence per bin.
    """
    num_bins = phases.shape[1]
    b = np.zeros(num_bins)
    for i in range(num_bins):
        h = np.histogram(
            phases[:, i],
            bins=10,
            range=(-np.pi, np.pi),
        )
        b[i] = bin_coherence(h[0])
    return b


def block_cplx_spectrum(
    blocks: np.ndarray,
    apply_ramp: bool = True,
    ramp_size: int = 0,
) -> npt.NDArray[np.complex128]:
    """Convert measurement blocks to complex spectra via rFFT.

    Args:
        blocks: 2D array of shape (n_blocks, n_samples).
        apply_ramp: If True, apply cosine-shaped ramps at
          start and end of each block.
        ramp_size: Length of the rising/falling edges in samples.
          Must be > 0 if apply_ramp is True.
    """
    blocks = np.asarray(blocks)
    if blocks.ndim != 2:
        raise ValueError("blocks must be a 2D array of shape (n_blocks, n_samples).")

    n_samples = blocks.shape[1]

    if apply_ramp:
        if ramp_size <= 0:
            raise ValueError("ramp_size must be > 0 when apply_ramp is True.")
        if 2 * ramp_size > n_samples:
            raise ValueError(
                "ramp_size is too large: 2 * ramp_size must be <= number of samples."
            )

        # Create cosine ramp from 0 -> 1
        t = np.linspace(0.0, np.pi, ramp_size, endpoint=True)
        rise = 0.5 - 0.5 * np.cos(t)    # 0 .. 1
        fall = rise[::-1]               # 1 .. 0

        window = np.ones(n_samples, dtype=float)
        window[:ramp_size] *= rise
        window[-ramp_size:] *= fall

        # Broadcast window over blocks
        blocks = blocks * window[np.newaxis, :]

    cplx_spectra = np.fft.rfft(blocks, axis=1)
    return cplx_spectra


def block_abs_spectrum(
    cplx_spectra: np.ndarray,
    num_block_samples: int
) -> np.ndarray:
    """Compute amplitude spectra from complex spectra."""
    return 2*np.abs(cplx_spectra) / num_block_samples
