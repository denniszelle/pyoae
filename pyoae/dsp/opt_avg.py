"""Module for optimized constructive averaging.

Description
-----------

Module that provides the `OptAverage` class with functions to obtain
a set of blocks the represent an optimized average, i.e., containing
only blocks that provide a gain to the averaging result.

Optimized averaging is a special type of sorted averaging. It
automatically excludes blocks compromised by artifacts or
excessive noise from the averaging process.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


NUM_MIN_OPT_AVG_BLOCKS = 3

@dataclass
class OptAvgStats:
    """Evaluation parameters of optimized averaging."""

    eval_fun: npt.NDArray[np.float32]
    """Noise evaluation function.

    1D array describing delta(M) with M as the number
    of included blocks sorted by their
    noise values.
    """

    gain_fun: npt.NDArray[np.float32]
    """Achieved gain of SNR due to averaging.

    1D array describing g_m(M) with M as the number
    of included blocks sorted by their
    noise values.
    """

    num_accepted_blocks: int
    """Number of accepted blocks by the optimized averager."""

    num_excluded_blocks: int
    """Number of excluded blocks by the optimized averager."""


class OptAverage:
    """Class to get indices for optimized averaging."""

    id: int
    """ID used as primary key for database entry."""

    noise_values: npt.NDArray[np.float32]
    """1D array with noise values for each block."""

    block_idx: npt.NDArray[np.int_]
    """1D array with block indices to be sorted."""

    i_received: int
    """Counter of received blocks."""

    accepted_idx: npt.NDArray[np.int_]
    """1D array with indices of accepted blocks."""

    stats: OptAvgStats
    """Container with evaluation parameters."""

    def __init__(self) -> None:
        """Initialize optimized averager instance."""
        super().__init__()
        self.noise_values = np.empty(0, np.float32)
        self.block_idx = np.empty(0, np.int_)
        self.i_received = 0
        self.accepted_idx = np.empty(0, np.int_)
        self.stats = OptAvgStats(
            np.empty(0, np.float32),
            np.empty(0, np.float32),
            0,
            0
        )

    def setup(self, num_max_blocks: int) -> None:
        """Reset the class"""
        self.noise_values = np.full(num_max_blocks, fill_value=np.inf)
        self.block_idx = np.arange(0, num_max_blocks, 1, dtype=np.int_)
        self.stats.eval_fun = np.zeros(num_max_blocks, dtype=np.float32)
        self.stats.gain_fun = np.zeros(num_max_blocks, dtype=np.float32)

    def evaluate_averaging(self) -> None:
        """Evaluate the averaging for the current data blocks."""
        if not self.noise_values.size:
            print('Evaluate averaging called without noise values.')
            return

        self.sort_noise_values()
        n1 = self.noise_values[0]

        if (
            self.i_received < NUM_MIN_OPT_AVG_BLOCKS
            or n1 == 0
        ):
            # At least three noise values required for optimized averaging.
            # Accept all blocks before reaching that limit.
            self.accepted_idx = np.arange(0, self.i_received, 1, dtype=np.int_)
            self.stats.num_accepted_blocks = self.i_received
            self.stats.num_excluded_blocks = 0
            if n1 == 0:
                print('Cannot evaluate averaging: baseline noise is zero.')
            return

        received_slice = slice(0, self.i_received)
        if np.any(np.isnan(self.noise_values[received_slice])):
            print('Invalid noise values (NaN) provided.')
        if np.any(self.noise_values[received_slice] == 0):
            print('Invalid noise values (0) provided.')

        delta_noise = self.noise_values - n1
        delta_noise_squared = delta_noise ** 2

        for m in range(2, self.i_received + 1):
            i_slice = slice(1, m-1)
            s = np.sum(
                2 * n1 * delta_noise[i_slice] + delta_noise_squared[i_slice]
            )

            r = np.sqrt(
                1 + n1**-2 * ((m/(m-1))*n1**2 + ((2*m-1)/((m-1)**2))*s)
            )
            self.stats.eval_fun[m-1] = n1 * (r - 1)
            self.stats.gain_fun[m-1] = np.sqrt(m / (1 + s/(m*n1**2)) )

        # Find boundary index
        is_accepted = delta_noise <= self.stats.eval_fun
        self.accepted_idx = self.block_idx[is_accepted]
        self.stats.num_accepted_blocks = len(self.accepted_idx)
        self.stats.num_excluded_blocks = self.i_received - self.stats.num_accepted_blocks

    def set_noise_value(self, pos: int, noise_val: float) -> None:
        """Receive a noise value for a block and increase counter.

        This increases the counter for received blocks.
        """
        self.noise_values[pos] = noise_val
        self.i_received += 1

    def sort_noise_values(self) -> None:
        """Sort the received blocks by their noise values."""
        received_slice = slice(0, self.i_received)
        sort_idx = np.argsort(self.noise_values[received_slice])

        sorted_noise = self.noise_values[received_slice][sort_idx]
        sorted_idx = self.block_idx[received_slice][sort_idx]

        self.noise_values[received_slice] = sorted_noise
        self.block_idx[received_slice] = sorted_idx
