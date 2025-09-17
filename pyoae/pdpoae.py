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
import scipy.signal

from pyoae import generator
from pyoae import get_logger
from pyoae import helpers
from pyoae.calib import MicroTransferFunction, OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.generator import PulseDpoaeStimulus
from pyoae.protocols import PulseDpoaeMsrmtParams
from pyoae.signals import PeriodicSignal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


@dataclass
class PulseDpoaePlotInfo:
    """Container with data for the measurement plot."""

    fig: Figure
    """Figure corresponding to the plot window."""

    axes: list[Axes]
    """Axis object of the time plot."""

    lines: list[Line2D]
    """Line object for the time plot."""

    update_interval: float
    """Interval to apply processing and plot update during measurement."""

    live_display_duration: float
    """Duration to display time domain plot in ms."""


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
    fs: float,
    block_size: int
) -> tuple[Figure, list[Axes], list[Line2D]]:
    """Sets up the plots.

    Args:
        recording_duration: Total duration of the recording in seconds
        fs: Sampling frequency in Hz
        block_size: Size of each measurement block that is repeated
          periodically
        live_display_duration: Time domain display duration in milliseconds
        is_calib_available: Boolean whether a calibration is available to
          display sound pressure or only show raw measurement data

    Returns:
        tuple[fig, line_time, line_spec]

        - **fig**: Object containing the plots
        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes: list[Axes]

    time_vector = np.arange(block_size, dtype=np.float32) / fs * 1E3
    block_duration = block_size / fs
    lines: list[Line2D] = []
    for axis in axes:

        y_wave = np.zeros_like(time_vector)
        line_time, = axis.plot(time_vector, y_wave)
        lines.append(line_time)
        axis.set_ylim(-1, 1)
        axis.set_xlim(0, block_duration * 1E3)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Amplitude (full scale)")

    axes[0].set_title("Recorded Segments")
    #axes[1].set_title("Recorded Ensemble")

    return fig, axes, lines


# def get_measurement_blocks(
#     recorded_signal: npt.NDArray[np.float32],
#     block_size: int
# ) -> npt.NDArray[np.float32]:
#     """Returns 2D array of measurement blocks."""
#     total_blocks = int(np.floor(len(recorded_signal)/block_size))
#     block_data = recorded_signal[:total_blocks*block_size]

#     # Reshape data into block structure and
#     # remove the first and last block
#     return block_data.reshape(-1, block_size)


def process_average(
    recorded_signal: npt.NDArray[np.float32],
    block_size: int,
    correction_tf: MicroTransferFunction | None,
    artifact_rejection_thr:float
) -> npt.NDArray[np.float32]:
    """Processes recorded signal to obtain spectrum.

    The signal is averaged in the time domain rejecting blocks
    above some value relative to the average RMS. Then, the
    spectrum is obtained from the averaged signal.

    In a typical setting evoking continuous DPOAE, the primary
    tones in the first and last acquisition block are multiplied
    with cosine-shaped fade-in and fade-out ramps, respectively.
    To avoid these ramps to influence the averaged spectrum, the
    first and last blocks are excluded from the overall averaging.


    Args:
        recorded_signal: float array of measurement data
        block_size: Size of each recording block in samples
        correction_tf: Transfer function of the microphone
        artifact_rejection_thr: Threshold representing a factor where blocks
          larger than factor*mean_rms are rejected

    Returns:
        Array of floats containing the averaged spectrum.
    """

    # Obtain an integer number of recorded blocks
    total_blocks = int(np.floor(len(recorded_signal)/block_size))
    block_data = recorded_signal[:total_blocks*block_size]
    # Only apply processing for at least 3 blocks
    avg = np.zeros(block_size)
    if total_blocks > 2:
        blocks = block_data.reshape(-1, block_size)

        # Reject RMS values larger than a scaled version
        # of the median RMS value.
        # rms_vals = np.sqrt(np.mean(np.square(blocks), axis=1))
        # median_rms = np.median(rms_vals)
        # accepted_idc = np.where(rms_vals<artifact_rejection_thr*median_rms)[0]

        # Average blocks
        # if len(accepted_idc):
        #     avg = blocks[accepted_idc].mean(axis=0)
        avg = blocks.mean(axis=0)

    return avg.astype(np.float32)


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

        if sync_msrmt.state is MsrmtState.FINISHED:
            # do not process average during measurement
            # TODO: perform computation on distinguished thread
            avg = process_average(
                recorded_signal,
                info.block_size,
                info.input_trans_fun,
                info.artifact_rejection_thr
            )

    return recorded_signal, avg


def update_plot_data(
    recorded_signal: npt.NDArray[np.float32],
    avg: npt.NDArray[np.float32],
    info: DpoaeUpdateInfo
) -> tuple[Line2D, Line2D]:
    """Updates the plot data.

    Create and set the plot objects for the data.

    Args:
        recorded_signal: Float array containing the raw signal.
        spectrum: Float array containing the spectral estimate.
        info: Info object containing meta infos and plot objects

    Returns:
        tuple[line_time, line_spec]

        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    if len(recorded_signal) < (info.block_size * (info.num_recorded_blocks + 1)):
        return info.plot_info.lines[0], info.plot_info.lines[1]

    #x_data = np.arange(info.block_size) / info.fs * 1E3

    # determine indices of current segment interval
    idx_start = info.num_recorded_blocks * info.block_size
    idx_end = (info.num_recorded_blocks + 1) * info.block_size
    info.num_recorded_blocks += 1
    # info.plot_info.lines[0].set_data(
    #     x_data,
    #     recorded_signal[idx_start:idx_end]
    # )
    info.plot_info.lines[0].set_ydata(
        recorded_signal[idx_start:idx_end]
    )
    info.plot_info.lines[1].set_ydata(avg)

    # Update y-axis limits.
    # avg_min = min(avg)
    # avg_max = max(avg)
    padding = 0.0 # padding on top and bottom

    rec_min = min(recorded_signal)
    rec_max = max(recorded_signal)

    rec_y_lim = info.plot_info.axes[0].get_ylim()
    if rec_min > rec_y_lim[0] or rec_max < rec_y_lim[1]:
        info.plot_info.axes[0].set_ylim(rec_min - padding, rec_max + padding)

    # if (
    #     avg_min < info.plot_info.axes[1].get_ylim()[0]
    #     or avg_max > info.plot_info.axes[1].get_ylim()[1]
    # ):
    #     info.plot_info.axes[1].set_ylim(avg_min - padding, avg_max + padding)

    return info.plot_info.lines[0], info.plot_info.lines[1]


def update_msrmt(
    frame,
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo
) -> tuple[Line2D, Line2D]:
    """Processes results from data and update the plots.

    Args:
        sync_msrmt: Measurement object that handles the synchronized
          measurement.
        info: Info object containing meta infos and plot objects

    Returns:
        tuple[line_time, line_spec]

        - **line_time**: Line object with the time-domain data of the signal
        - **line_spec**: Line object with the spectral data of the signal
    """

    del frame

    if sync_msrmt.state == MsrmtState.FINISHED:
        # plt.close(info.fig)
        return info.plot_info.lines[0], info.plot_info.lines[1]

    if sync_msrmt.state == MsrmtState.FINISHING:
        sync_msrmt.state = MsrmtState.FINISHED

    # recorded_signal, avg = get_results(sync_msrmt, info)
    return info.plot_info.lines[0], info.plot_info.lines[1]
    #return update_plot_data(recorded_signal, avg, info)


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


def plot_offline(
    sync_msrmt: SyncMsrmt,
    info: DpoaeUpdateInfo,
    stimulus: PulseDpoaeStimulus
) -> None:
    """Plots the final results in a non-updating plot.

    This function obtains the results from the measurement object, creates a
    plot and shows the complete measurement as well as the spectral estimate.
    """
    if sync_msrmt.state != MsrmtState.FINISHED:
        return

    # has_input_calib =  info.input_trans_fun is not None

    recorded_signal, avg = get_results(sync_msrmt, info)
    _, axes, lines = setup_plot(
        sync_msrmt.recording_data.fs,
        info.block_size
    )

    # plot recording overview
    lines[0].set_xdata(np.arange(len(recorded_signal))/info.fs)
    lines[0].set_ydata(recorded_signal)
    axes[0].set_xlim(0, sync_msrmt.recording_data.msrmt_duration)
    axes[0].set_xlabel("Recording Time (s)")


    # rec_min = min(recorded_signal)
    # rec_max = max(recorded_signal)
    padding = 0  # padding on top and bottom


    # filter signal and convert to muPa
    fdp = 2*stimulus.f1 - stimulus.f2
    if info.input_trans_fun is not None:
        # TODO: use FFT for unfiltered signals?
        s = info.input_trans_fun.get_sensitivity(fdp)
        avg /= s  # convert to muPa

    avg = avg - np.mean(avg)

    hp_order = 525
    hp_delay = int(0.5*(hp_order-1))

    b_hp = scipy.signal.firwin(
        hp_order,
        200,
        pass_zero='highpass',  # type: ignore
        fs=DeviceConfig.sample_rate
    )
    avg_hp = np.zeros_like(avg)
    avg_hp[:-hp_delay] = scipy.signal.lfilter(b_hp, 1, avg)[hp_delay:]

    # perform simple band-pass filtering
    t_hw_sp = generator.short_pulse_half_width(stimulus.f2) * 1E-3
    bw = 0.5 * (1 / t_hw_sp)
    df = int(0.5*bw)
    cutoff = fdp + np.array([-df, df], dtype=np.float32)
    bp_order = 1025
    b = scipy.signal.firwin(
        bp_order,
        cutoff,
        pass_zero='bandpass',  # type: ignore
        fs=DeviceConfig.sample_rate
    )

    # add ramp
    ramp_len = int(
        DeviceConfig.ramp_duration * 1E-3 * DeviceConfig.sample_rate
    )
    ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
    ramp = ramp.astype(np.float32)
    win = np.ones_like(avg)
    win[:ramp_len] = ramp
    win[-ramp_len:] = ramp[::-1]
    avg_hp *= win

    bp_delay = int(0.5*(bp_order-1))
    avg_bp = np.zeros_like(avg)
    avg_bp[:-bp_delay] = scipy.signal.lfilter(b, 1, avg_hp)[bp_delay:]

    avg_min = min(avg_bp)
    avg_max = max(avg_bp)
    axes[1].set_ylim(avg_min - padding, avg_max + padding)
    lines[1].set_ydata(avg_bp)
    plt.tight_layout()
    plt.show()


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
        plot_offline(self.msrmt, self.update_info, self.stimulus)

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
        recorded_signal, _ = get_results(self.msrmt, self.update_info)
        np.savez(save_path,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate,
            f1=self.stimulus.f1,
            f2=self.stimulus.f2,
            level1=self.stimulus.level1,
            level2=self.stimulus.level2,
            num_block_samples=self.update_info.block_size,
            recorded_sync=self.msrmt.live_msrmt_data.sync_recorded
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
        num_block_samples: int,
    ) -> PulseDpoaePlotInfo:
        """Sets up live plot and measurement information."""
        fig, axes, lines = setup_plot(
            DeviceConfig.sample_rate,
            num_block_samples
        )
        return PulseDpoaePlotInfo(
            fig,
            axes,
            lines,
            DeviceConfig.update_interval,
            DeviceConfig.live_display_duration
        )
