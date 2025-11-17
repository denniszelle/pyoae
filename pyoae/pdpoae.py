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

from datetime import datetime
from logging import Logger
import os

from matplotlib import pyplot as plt
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
from pyoae.dsp.processing import DpoaeMsrmtData, PulseDpoaeProcessor
from pyoae.generator import PulseDpoaeStimulus
from pyoae.plot_context import PlotContext
from pyoae.msrmt_context import DpoaeMsrmtContext
from pyoae.protocols import PulseDpoaeMsrmtParams
from pyoae.signals import PeriodicSignal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


logger = get_logger()


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
    msrmt_ctx: DpoaeMsrmtContext
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Processes data and returns plot results.

    If the measurement is currently running, the recorded signal
    is obtained and a synchronously averaged spectrum is estimated.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.

    Returns:
        tuple[recorded_signal, spectrum]

        - **recorded_signal**: Float array with the recorded signal
        - **spectrum**: Float array with the spectral estimate
    """

    # Only update while or after main measurement
    avg = np.zeros(msrmt_ctx.block_size, np.float32)
    recorded_signal = np.zeros(msrmt_ctx.block_size, np.float32)
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
    msrmt_ctx: DpoaeMsrmtContext,
    plot_ctx: PlotContext
) -> list[Line2D]:
    """Updates the plot data.

    Sets the plot data using line objects.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.
        plot_ctx: Parameters and instances to control plots.

    Returns:
        list[line]

    """
    recorded_signal = sync_msrmt.get_recorded_signal()
    num_recorded_samples = len(recorded_signal)
    num_recorded_blocks = num_recorded_samples // msrmt_ctx.block_size

    if num_recorded_blocks <= msrmt_ctx.num_recorded_blocks:
        return [plot_ctx.line]

    # x_data = np.arange(num_recorded_samples) / info.fs
    idx_start = (num_recorded_blocks - 1) * msrmt_ctx.block_size
    idx_stop = idx_start + msrmt_ctx.block_size
    cur_block = recorded_signal[idx_start:idx_stop]
    plot_ctx.line.set_ydata(
        cur_block
    )
    msrmt_ctx.num_recorded_blocks = num_recorded_blocks
    # Update y-axis limits on demand.
    padding = 0.05 # padding on top and bottom
    rec_min = min(cur_block) - padding
    rec_max = max(cur_block) + padding
    rec_min = np.floor(rec_min / 0.05) * 0.05
    rec_max = np.ceil(rec_max / 0.05) * 0.05

    rec_y_lim = plot_ctx.axes.get_ylim()
    if rec_min != rec_y_lim[0] or rec_max != rec_y_lim[1]:
        plot_ctx.axes.set_ylim(rec_min, rec_max)
    # info.plot_info.axes.set_xlim(x_data[0], x_data[1] + (1/info.fs) * 1E3)
    return [plot_ctx.line]


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    msrmt_ctx: DpoaeMsrmtContext,
    plot_ctx: PlotContext
) -> list[Line2D]:
    """Processes results from data and update the plots.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        msrmt_ctx: Parameters and instances to control the measurement.
        plot_ctx: Parameters and instances to control plots.

    Returns:
        list[Line2D]

        - **line_time**: Line object with the time-domain data of the signal
    """

    del frame

    if sync_msrmt.state == MsrmtState.FINISHED:
        if msrmt_ctx.msrmt_anim is not None and msrmt_ctx.non_interactive:
            msrmt_ctx.msrmt_anim.stop_animation()
        return [plot_ctx.line]

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.set_state(MsrmtState.FINISHED)
        if msrmt_ctx.non_interactive:
            logger.info('Recording complete.')
        else:
            logger.info('Recording complete. Please close window to continue.')

    return update_plot_data(sync_msrmt, msrmt_ctx, plot_ctx)


class PulseDpoaeRecorder:
    """Class to manage a DPOAE recording."""

    stimulus: PulseDpoaeStimulus
    """Parameters of primary tones."""

    signals: list[PeriodicSignal]
    """List of output signals for each channel."""

    plot_ctx: PlotContext
    """Instance to plot context for SOAE visualization."""

    msrmt_ctx: DpoaeMsrmtContext
    """Instance to context to control SOAE measurement updates."""

    msrmt: SyncMsrmt
    """Instance to perform a synchronized OAE measurement."""

    dpoae_processor: PulseDpoaeProcessor | None
    """Dpoae processor for offline post-processing"""

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
        non_interactive: bool = False,
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

        # stimulus will be set during `generate_output_signals``
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
        self.plot_ctx = self.setup_plot_context(num_block_samples)
        self.msrmt_ctx = DpoaeMsrmtContext(
            fs=DeviceConfig.sample_rate,
            block_size=num_block_samples,
            input_trans_fun=mic_trans_fun,
            artifact_rejection_thr=DeviceConfig.artifact_rejection_threshold,
            non_interactive=non_interactive,
            msrmt_anim=None,
            f1=self.stimulus.f1,
            f2=self.stimulus.f2,
            num_recorded_blocks=0
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
        self.dpoae_processor = None

    def record(self) -> None:
        """Starts the recording."""
        self.logger.info("Starting recording...")
        # `start_msrmt` starts the application loop
        self.msrmt.start_msrmt(update_msrmt, self.msrmt_ctx, self.plot_ctx)

        # Plot all data and final result after user has
        # closed the live-measurement window.

        # We utilize the `PulseDpoaeProcessor`, which can handle
        # both raw data from files as well as from the recorder.
        recorded_signal = self.msrmt.get_recorded_signal()
        if not recorded_signal.size:
            return
        recording: DpoaeMsrmtData = {
            'recorded_signal': recorded_signal,
            'samplerate': DeviceConfig.sample_rate,
            'f1': self.stimulus.f1,
            'f2': self.stimulus.f2,
            'level1': self.stimulus.level1,
            'level2': self.stimulus.level2,
            'num_block_samples': self.msrmt_ctx.block_size,
            'recorded_sync': self.msrmt.live_msrmt_data.sync_recorded
        }
        self.dpoae_processor = PulseDpoaeProcessor(
            recording, self.msrmt_ctx.input_trans_fun
        )
        self.dpoae_processor.process_msrmt()
        if not self.msrmt_ctx.non_interactive:
            self.logger.info(
                'Showing offline results. Please close window to continue.'
            )
            self.dpoae_processor.plot()

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
            str(int(self.stimulus.f2)),
            str(int(self.stimulus.level2)),
        ]
        file_name = "_".join(filter(None, parts))
        save_path = os.path.join(save_path, file_name)
        recorded_signal, _ = get_results(self.msrmt, self.msrmt_ctx)
        if self.dpoae_processor is not None:
            raw_avg = self.dpoae_processor.raw_averaged
            avg = self.dpoae_processor.dpoae_signal
        else:
            raw_avg = np.array(0,np.float64)
            avg = np.array(0,np.float64)
        np.savez(save_path,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate,
            f1=self.stimulus.f1,
            f2=self.stimulus.f2,
            level1=self.stimulus.level1,
            level2=self.stimulus.level2,
            num_block_samples=self.msrmt_ctx.block_size,
            recorded_sync=self.msrmt.live_msrmt_data.sync_recorded,
            average=avg,
            raw_average=raw_avg
        )
        self.logger.info("Measurement saved to %s.npz", save_path)

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

    def setup_plot_context(
        self,
        block_size: int
    ) -> PlotContext:
        """Sets up live plot and measurement information."""
        fig, axes, line = setup_plot(
            block_size,
            DeviceConfig.sample_rate
        )
        update_interval = (block_size / DeviceConfig.sample_rate) * 1E3
        return PlotContext(
            fig=fig,
            axes=axes,
            line=line,
            update_interval=update_interval,
            live_display_duration=update_interval
        )
