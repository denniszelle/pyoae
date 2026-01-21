"""Classes to implement plotting in a different process"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.sharedctypes import Synchronized
import numpy as np
from typing import Any

from multiprocessing.synchronize import Event

from pyoae.device.device_config import DeviceConfig


@dataclass
class MsrmtEvents:
    """Events used for measurement sequence"""

    enable_plot: Event
    """Event that enables the plot when set.

    This prevents plotting while syncing
    """

    cancel_msrmt: Event
    """Event to cancel the measurement

    This triggers a measurement cancel when the plotting window is closed"""


class ProcessPlotter:
    """Plot in a separate process."""
    def __init__(
        self,
        record_idx_share: Synchronized,
        num_samples: int,
        fs: float,
        msrmt_events: MsrmtEvents,
        interval_length: float=0.08,
    ):
        self.record_idx_share = record_idx_share
        self.num_samples = num_samples
        self.fs = fs
        self.display_samples = int(interval_length*fs)
        self.time_vec = np.arange(self.display_samples) / self.fs *1E3
        self.running = True
        self.display_interval = interval_length  # in seconds
        self.msrmt_events = msrmt_events

        self.data = np.ndarray((0,), dtype=np.float32)
        self.shm = None
        self.fig = None
        self.line = None

    def terminate(self):
        """Terminate the plot window"""
        print('Terminating figure.')
        self.running = False
        plt.close('all')

    def update_plot(self):
        """Update the current plot"""

        if self.line is None or self.fig is None:
            return True

        if not self.msrmt_events.enable_plot.is_set():
            return True

        with self.record_idx_share.get_lock():
            record_idx = self.record_idx_share.value

        last_full_block_idx = (record_idx // self.display_samples) * self.display_samples
        if record_idx > self.display_samples:
            interval = [
                last_full_block_idx-self.display_samples,
                last_full_block_idx
            ]
            time_vec = None
        else:
            interval = [0, record_idx]
            time_vec = np.arange(record_idx)/self.fs*1E3
        try:
            y = self.data[interval[0]:interval[1]]
            if time_vec is None:
                self.line.set_data(self.time_vec, y)
            else:
                self.line.set_data(time_vec, y)
            self.fig.canvas.draw_idle()
        except (IndexError, ValueError, RuntimeError) as e:
            print('Plotting error: ', e)
            self.terminate()
            return False
        return self.running

    def run(self, shm_name: str):
        """Run plot process"""
        self.shm = shared_memory.SharedMemory(name=shm_name)

        self.data = np.ndarray((self.num_samples,), dtype=np.float32, buffer=self.shm.buf)

        self.fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        self.line, = ax.plot([], [])
        ax.set_xlim(0, self.display_samples/self.fs*1E3)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title('Recorded Wavefcorm')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (full scale)')

        self.fig.canvas.mpl_connect(
            'close_event', self._on_close
        )

        time = self.fig.canvas.new_timer(
            interval=int(DeviceConfig.update_interval)
        )
        time.add_callback(self.update_plot)
        time.start()

        plt.show()
        self.terminate()
        # self.shm.close()

    def _on_close(self, _):
        self.msrmt_events.cancel_msrmt.set()


class LivePlotProcess:
    """Controller for live plotting."""
    def __init__(
        self,
        shm_name: str,
        record_idx_share: Any,
        num_samples: int,
        fs: float,
        msrmt_events: MsrmtEvents,
        interval: float=0.08
    ):
        self.plotter = ProcessPlotter(
            record_idx_share,
            num_samples,
            fs,
            msrmt_events,
            interval
        )
        self.plot_process = mp.Process(
            target=self.plotter.run,
            args=(shm_name,),
            daemon=True
        )
        self.plot_process.start()

    def stop(self):
        """Stop process if alive"""
        if self.plot_process.is_alive():
            self.plot_process.terminate()
