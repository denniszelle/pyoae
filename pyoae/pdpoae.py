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

import numpy as np
import numpy.typing as npt

from pyoae import generator
from pyoae import get_logger
from pyoae import helpers
from pyoae.calib import MicroTransferFunction, OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.dsp.containers import DpoaeMsrmtData
from pyoae.dsp.processing import PulseDpoaeProcessor
from pyoae.generator import PulseDpoaeStimulus
from pyoae.msrmt_context import DpoaeMsrmtContext
from pyoae.protocols import PulseDpoaeMsrmtParams
from pyoae.signals import PeriodicSignal
from pyoae.sync import HardwareData, RecordingData, SyncMsrmt, MsrmtState


logger = get_logger()


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


class PulseDpoaeRecorder:
    """Class to manage a DPOAE recording."""

    stimulus: PulseDpoaeStimulus
    """Parameters of primary tones."""

    signals: list[PeriodicSignal]
    """List of output signals for each channel."""

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
        recording_duration = (
            num_total_recording_samples / DeviceConfig.sample_rate
        )

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
        self.msrmt_ctx = DpoaeMsrmtContext(
            fs=DeviceConfig.sample_rate,
            block_size=num_block_samples,
            input_trans_fun=mic_trans_fun,
            non_interactive=non_interactive,
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
            self.signals,
            block_duration
        )
        self.dpoae_processor = None

    def record(self) -> None:
        """Starts the recording."""
        self.logger.info("Starting recording...")

        self.msrmt.run_msrmt()

        # Plot all data and final result measurement has finished.

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
