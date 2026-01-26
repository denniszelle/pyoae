"""Classes and functions to record continuous DPOAEs

This module contains utility functions used to acquire, process
and analyze continuous distortion-product otoacoustic emissions (cDPOAE).
These functions are typically used within example scripts, but can
also be imported and reused in other analysis pipelines.

Key functionalities include:
- Setup of live plots for the measurement
- Robust, synchronous spectrum estimation

Typical usage:

```
    from pyoae.cdpoae import DpoaeRecorder
    dpoae_recorder = DpoaeRecorder(msrmt_params)
    dpoae_recorder.record()
```

This module is not intended to be run directly.
"""

from datetime import datetime
from logging import Logger
import os

import numpy as np

from pyoae import generator
from pyoae import get_logger
from pyoae import helpers
from pyoae.calib import MicroTransferFunction, OutputCalibration
from pyoae.device.device_config import DeviceConfig
from pyoae.dsp.containers import DpoaeMsrmtData
from pyoae.dsp.processing import ContDpoaeProcessor
from pyoae.generator import ContDpoaeStimulus
from pyoae.msrmt_context import DpoaeMsrmtContext
from pyoae.protocols import DpoaeMsrmtParams
from pyoae.signals import PeriodicRampSignal
from pyoae.sync import (
    HardwareData,
    RecordingData,
    SyncMsrmt
)


logger = get_logger()


class DpoaeRecorder:
    """Class to manage a DPOAE recording."""

    stimulus: ContDpoaeStimulus
    """Parameters of primary tones."""

    signals: list[PeriodicRampSignal]
    """List of output signals for each channel."""

    msrmt_ctx: DpoaeMsrmtContext
    """Instance to context to control DPOAE measurement updates."""

    dpoae_processor: ContDpoaeProcessor | None
    """Dpoae processor for offline post-processing"""

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
        msrmt_params: DpoaeMsrmtParams,
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
            msrmt_params['num_averaging_blocks'] * num_block_samples
        )
        block_duration = num_block_samples / DeviceConfig.sample_rate
        recording_duration = num_total_recording_samples / DeviceConfig.sample_rate

        if mic_trans_fun:
            mic_trans_fun.num_samples = num_block_samples
            mic_trans_fun.sample_rate = DeviceConfig.sample_rate
            mic_trans_fun.interpolate_transfer_fun()

        if block_duration != msrmt_params["block_duration"]:
            self.logger.warning(
                'Block duration adjusted to %.2f ms',
                block_duration * 1E3
            )

        self.stimulus = ContDpoaeStimulus(
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

        # Plot all data and final result after user has
        # closed the live-measurement window.

        # We utilize the `ContDpoaeProcessor` to handle
        # raw data from the recorder.
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

        self.dpoae_processor = ContDpoaeProcessor(
            recording,
            mic=self.msrmt_ctx.input_trans_fun
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
            "cdpoae_msrmt",
            time_stamp,
            helpers.sanitize_filename_part(self.subject),
            helpers.sanitize_filename_part(self.ear),
            str(int(self.stimulus.f2)),
            str(int(self.stimulus.level2)),
        ]
        file_name = "_".join(filter(None, parts))
        save_path = os.path.join(save_path, file_name)
        recorded_signal = self.msrmt.get_recorded_signal()
        if self.dpoae_processor is not None:
            averaged = self.dpoae_processor.raw_averaged
            spectrum = self.dpoae_processor.dpoae_spectrum
        else:
            averaged = np.array(0,np.float64)
            spectrum = np.array(0,np.float64)
        np.savez(save_path,
            average=averaged,
            spectrum=spectrum,
            recorded_signal=recorded_signal,
            samplerate=DeviceConfig.sample_rate,
            f1=self.stimulus.f1,
            f2=self.stimulus.f2,
            level1=self.stimulus.level1,
            level2=self.stimulus.level2,
            num_block_samples = self.msrmt_ctx.block_size,
            recorded_sync=self.msrmt.live_msrmt_data.sync_recorded
        )
        self.logger.info("Saved measurement to %s.npz", save_path)

    def generate_output_signals(
        self,
        msrmt_params: DpoaeMsrmtParams,
        block_duration: float,
        num_block_samples: int,
        num_total_recording_samples: int,
        out_calib: OutputCalibration | None = None
    ) -> None:
        """Generates the output signals for playback."""
        self.stimulus.calculate_frequencies(
            msrmt_params,
            block_duration
        )
        self.stimulus.level1 = generator.calculate_pt1_level(msrmt_params)
        self.stimulus.level2 = msrmt_params["level2"]
        stimulus1, stimulus2 = self.stimulus.generate_stimuli(
            num_block_samples,
            output_calibration=out_calib
        )

        # we always use rising and falling edges
        ramp_len = int(
            DeviceConfig.ramp_duration * 1E-3 * DeviceConfig.sample_rate
        )
        ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
        ramp = ramp.astype(np.float32)

        signal1 = PeriodicRampSignal(
            stimulus1,
            num_total_recording_samples,
            ramp
        )
        signal2 = PeriodicRampSignal(
            stimulus2,
            num_total_recording_samples,
            ramp
        )

        self.signals.append(signal1)
        self.signals.append(signal2)
