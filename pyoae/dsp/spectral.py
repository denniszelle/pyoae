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
