"""Classes and functions to record pulsed DPOAEs

This module contains utility functions used to acquire pulsed
distortion-product otoacoustic emissions (pDPOAE).
These functions are typically used within recording scripts, but can
also be imported and reused in other recording pipelines.

Key functionalities include:
- Setup of plots for the measurement
- Simplified averaging

Typical usage:

```
    from pyoae.pdpoae import PulseDpoaeRecorder
    dpoae_recorder = PulseDpoaeRecorder(msrmt_params)
    dpoae_recorder.record()
```

This module is not intended to be run directly.
"""

from dataclasses import dataclass
from datetime import datetime
from logging import Logger
import os

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt

from pyoae import generator
from pyoae import get_logger
from pyoae import helpers
from pyoae.calib import MicroTransferFunction, OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.dsp.processing import PulseDpoaeMsrmtData, PulseDpoaeProcessor
from pyoae.generator import PulseDpoaeStimulus
from pyoae.protocols import PulseDpoaeMsrmtParams
from pyoae.signals import PeriodicSignal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


logger = get_logger()


@dataclass
class PulseDpoaePlotInfo:
    """Container with data for the measurement plot."""

    fig: Figure
    """Figure corresponding to the plot window."""

    axes: Axes
    """Axes object of the online measurement plot."""

    line: Line2D
    """Line object for the online measurement plot."""

    update_interval: float
    """Interval to apply processing and plot update during measurement."""


@dataclass
class DpoaeUpdateInfo:
    """Container with data for continuous DPOAE measurement updates."""

    plot_info: PulseDpoaePlotInfo
    """Storage for the measurement plot."""

    fs: float
    """Sampling frequency in Hz."""

    block_size: int
    """Number of samples in each block."""

    num_recorded_blocks: int
    """Number of recorded blocks."""

    artifact_rejection_thr: float
    """Threshold for simple artifact rejection.

    Reject blocks with a root-mean-square (RMS) value
    exceeding ARTIFACT_REJECTION_THR * median_rms.
    """

    input_trans_fun: MicroTransferFunction | None = None
    """Handle to microphone transfer function.

    A microphone transfer function is used to correct the
    recorded signal from the microphone characteristics.

    Note:
        This is a dummy object for future implementation.
    """


def setup_plot(
    block_size: int,
    fs: float,
) -> tuple[Figure, Axes, Line2D]:
    """Sets up the plots.

    Args:
        recording_duration: Total duration of the recording in seconds
        fs: Sampling frequency in Hz
        block_size: Size of each measurement block that is repeated
          periodically
        is_calib_available: Boolean whether a calibration is available to
          display sound pressure or only show raw measurement data

    Returns:
        tuple[fig, line_time, line_spec]

        - **fig**: Object containing the plots
        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Set up time plot
    x_wave = np.arange(block_size, dtype=np.float32) / fs
    y_wave = np.zeros_like(x_wave)
    line_time, = ax.plot(
        x_wave,
        y_wave
    )
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, x_wave[-1])
    ax.set_title("Recorded Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (full scale)")

    return fig, ax, line_time


def get_results(
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Processes data and returns plot results.

    If the measurement is currently running, the recorded signal
    is obtained and a synchronously averaged spectrum is estimated.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos about the measurement.

    Returns:
        tuple[recorded_signal, spectrum]

        - **recorded_signal**: Float array with the recorded signal
        - **spectrum**: Float array with the spectral estimate
    """

    # Only update while or after main measurement
    avg = np.zeros(info.block_size, np.float32)
    recorded_signal = np.zeros(info.block_size, np.float32)
    if sync_msrmt.state in [
        MsrmtState.RECORDING,
        MsrmtState.END_RECORDING,
        MsrmtState.FINISHING,
        MsrmtState.FINISHED
    ]:
        recorded_signal = sync_msrmt.get_recorded_signal()

        # if sync_msrmt.state is MsrmtState.FINISHED:
        #     # do not process average during measurement

    return recorded_signal, avg


def update_plot_data(
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo
) -> list[Line2D]:
    """Updates the plot data.

    Sets the plot data using line objects.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos about the measurement.

    Returns:
        list[line]

    """
    recorded_signal = sync_msrmt.get_recorded_signal()
    num_recorded_samples = len(recorded_signal)
    num_recorded_blocks = num_recorded_samples // info.block_size

    if num_recorded_blocks <= info.num_recorded_blocks:
        return [info.plot_info.line]

    # x_data = np.arange(num_recorded_samples) / info.fs
    idx_start = (num_recorded_blocks - 1) * info.block_size
    idx_stop = idx_start + info.block_size
    cur_block = recorded_signal[idx_start:idx_stop]
    info.plot_info.line.set_ydata(
        cur_block
    )
    info.num_recorded_blocks = num_recorded_blocks
    # Update y-axis limits on demand.
    padding = 0.05 # padding on top and bottom
    rec_min = min(cur_block) - padding
    rec_max = max(cur_block) + padding
    rec_min = np.floor(rec_min / 0.05) * 0.05
    rec_max = np.ceil(rec_max / 0.05) * 0.05

    rec_y_lim = info.plot_info.axes.get_ylim()
    if rec_min != rec_y_lim[0] or rec_max != rec_y_lim[1]:
        info.plot_info.axes.set_ylim(rec_min, rec_max)
    # info.plot_info.axes.set_xlim(x_data[0], x_data[1] + (1/info.fs) * 1E3)
    return [info.plot_info.line]


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo
) -> list[Line2D]:
    """Processes results from data and update the plots.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos and plot objects

    Returns:
        list[Line2D]

        - **line_time**: Line object with the time-domain data of the signal
    """

    del frame

    if sync_msrmt.state == MsrmtState.FINISHED:
        # plt.close(info.fig)
        return [info.plot_info.line]

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.set_state(MsrmtState.FINISHED)
        logger.info('Recording complete. Please close window to continue.')

    return update_plot_data(sync_msrmt, info)


def start_plot(sync_msrmt: SyncMsrmt, info: DpoaeUpdateInfo) -> None:
    """Executes the measurement plot that is regularly updated."""
    _ = FuncAnimation(
        info.plot_info.fig,
        update_msrmt,
        fargs=(sync_msrmt, info,),
        interval=info.plot_info.update_interval,
        blit=False,
        cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()
    if sync_msrmt.state not in [MsrmtState.FINISHING, MsrmtState.FINISHED]:
        sync_msrmt.state = MsrmtState.CANCELED


class PulseDpoaeRecorder:
    """Class to manage a DPOAE recording."""

    stimulus: PulseDpoaeStimulus
    """Parameters of primary tones."""

    signals: list[PeriodicSignal]
    """List of output signals for each channel."""

    update_info: DpoaeUpdateInfo
    """Instance to control DPOAE measurement updates."""

    msrmt: SyncMsrmt
    """Instance to perform a synchronized OAE measurement."""

    subject: str
    """Name/ID of the subject to be used for the measurement file name."""

    ear: str
    """Recording ear (left/right) to be used for the measurement file name."""

    logger: Logger
    """Class logger for debug, info, warning, and error messages."""

    def __init__(
        self,
        msrmt_params: PulseDpoaeMsrmtParams,
        mic_trans_fun: MicroTransferFunction | None = None,
        out_trans_fun: OutputCalibration | None = None,
        subject: str = '',
        ear: str = '',
        log: Logger | None = None
    ) -> None:
        """Creates a DPOAE recorder for given measurement parameters."""
        self.logger = log or get_logger()
        self.subject = subject
        self.ear = ear
        num_block_samples = int(
            msrmt_params['block_duration'] * DeviceConfig.sample_rate
        )
        num_total_recording_samples = (
            msrmt_params['num_averaging_blocks'] * num_block_samples * 4
        )
        block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = num_total_recording_samples / DeviceConfig.sample_rate

        if mic_trans_fun:
            mic_trans_fun.num_samples = num_block_samples
            mic_trans_fun.sample_rate = DeviceConfig.sample_rate
            mic_trans_fun.interpolate_transfer_fun()

        if block_duration != msrmt_params["block_duration"]:
            self.logger.warning(
                'Block duration adjusted to %.2f ms.',
                block_duration * 1E3
            )

        self.stimulus = PulseDpoaeStimulus(
            f1=0.0,
            f2=0.0,
            level1=0.0,
            level2=0.0
        )
        self.signals = []
        self.generate_output_signals(
            msrmt_params,
            block_duration,
            num_block_samples,
            num_total_recording_samples,
            out_calib=out_trans_fun
        )
        # has_input_calib = mic_trans_fun is not None
        dpoae_info = self.setup_info(num_block_samples)
        ARTIFACT_REJ_RATIO = 1.8  # TODO: replace
        self.update_info = DpoaeUpdateInfo(
            dpoae_info,
            DeviceConfig.sample_rate,
            num_block_samples,
            num_recorded_blocks=0,
            artifact_rejection_thr=ARTIFACT_REJ_RATIO,
            input_trans_fun=mic_trans_fun
        )
        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_block_samples,
            DeviceConfig.device_buffer_size
        )
        hw_data = HardwareData(
            2,
            2,
            DeviceConfig.input_device,
            DeviceConfig.output_device,
        )
        self.msrmt = SyncMsrmt(
            rec_data,
            hw_data,
            self.signals
        )

    def record(self) -> None:
        """Starts the recording."""
        self.logger.info("Starting recording...")
        # `start_msrmt` starts the application loop
        self.msrmt.start_msrmt(start_plot, self.update_info)

        # Plot all data and final result after user has
        # closed the live-measurement window.

        # We utilize the `PulseDpoaeProcessor`, which can handle
        # both raw data from files as well as from the recorder.
        recorded_signal = self.msrmt.get_recorded_signal()
        if not recorded_signal.size:
            return
        recording: PulseDpoaeMsrmtData = {
            'recorded_signal': recorded_signal,
            'samplerate': DeviceConfig.sample_rate,
            'f1': self.stimulus.f1,
            'f2': self.stimulus.f2,
            'level1': self.stimulus.level1,
            'level2': self.stimulus.level2,
            'num_block_samples': self.update_info.block_size,
            'recorded_sync': self.msrmt.live_msrmt_data.sync_recorded
        }
        p = PulseDpoaeProcessor(recording, self.update_info.input_trans_fun)
        p.process_msrmt()
        self.logger.info(
            'Showing offline results. Please close window to continue.'
        )
        p.plot()

    def save_recording(self) -> None:
        """Stores the measurement data in binary file."""
        save_path = os.path.join(
            os.getcwd(),
            'measurements'
        )
        os.makedirs(save_path, exist_ok=True)
        cur_time = datetime.now()
        time_stamp = cur_time.strftime("%y%m%d-%H%M%S")
        parts = [
            "pdpoae_msrmt",
            time_stamp,
            helpers.sanitize_filename_part(self.subject),
            helpers.sanitize_filename_part(self.ear),
            str(self.stimulus.f2),
            str(self.stimulus.level2),
        ]
        file_name = "_".join(filter(None, parts))
        save_path = os.path.join(save_path, file_name)
        recorded_signal, avg = get_results(self.msrmt, self.update_info)
        np.savez(save_path,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate,
            f1=self.stimulus.f1,
            f2=self.stimulus.f2,
            level1=self.stimulus.level1,
            level2=self.stimulus.level2,
            num_block_samples=self.update_info.block_size,
            recorded_sync=self.msrmt.live_msrmt_data.sync_recorded,
            average=avg
        )
        self.logger.info("Measurement saved to %s.", save_path)

    def generate_output_signals(
        self,
        msrmt_params: PulseDpoaeMsrmtParams,
        block_duration: float,
        num_block_samples: int,
        num_total_recording_samples: int,
        out_calib: OutputCalibration | None = None
    ) -> None:
        """Generates the output signals for playback."""
        self.stimulus.calculate_frequencies(msrmt_params)
        self.stimulus.level1 = generator.calculate_pt1_level(msrmt_params)
        self.stimulus.level2 = msrmt_params["level2"]
        self.stimulus.create_stimulus_mask(block_duration, msrmt_params)
        stimulus1, stimulus2 = self.stimulus.generate_stimuli(
            num_block_samples,
            output_calibration=out_calib
        )

        signal1 = PeriodicSignal(
            stimulus1,
            num_total_recording_samples,
        )
        signal2 = PeriodicSignal(
            stimulus2,
            num_total_recording_samples,
        )

        self.signals.append(signal1)
        self.signals.append(signal2)

    def setup_info(
        self,
        block_size: int,
    ) -> PulseDpoaePlotInfo:
        """Sets up live plot and measurement information."""
        fig, axes, lines = setup_plot(
            block_size,
            DeviceConfig.sample_rate
        )
        return PulseDpoaePlotInfo(
            fig,
            axes,
            lines,
            #DeviceConfig.update_interval * 4  # NUM_PTPV
            (block_size / DeviceConfig.sample_rate) * 1E3
        )
