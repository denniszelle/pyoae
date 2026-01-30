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

from dataclasses import dataclass
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
from pyoae.signals import PeriodicRampSignal, ZeroSignal
from pyoae.sync import (
    get_input_channels,
    HardwareData,
    RecordingData,
    SyncMsrmt,
)


logger = get_logger()


@dataclass
class ContDpoaeMsrmtInfo:
    """Information of a continuous DPOAE measurement"""

    stimulus: ContDpoaeStimulus
    """Parameters of primary tones."""

    msrmt_ctx: DpoaeMsrmtContext
    """Instance of context to control DPOAE measurement updates"""

    dpoae_processor: ContDpoaeProcessor | None
    """DPOAE processor for offline post-processing"""

    ear: str
    """Recording ear (left/right) to be used for the measurement file name."""

    num_total_recording_samples: int
    """Total number of recording samples for measurement"""

    num_block_samples: int
    """Number of samples per measurement block"""


class DpoaeRecorder:
    """Class to manage a DPOAE recording."""

    msrmt_info: list[ContDpoaeMsrmtInfo]
    """Information of measurements"""

    signals: list[PeriodicRampSignal | ZeroSignal]
    """List of output signals for each channel."""

    msrmt: SyncMsrmt | None
    """Instance to perform a synchronized OAE measurement."""

    subject: str
    """Name/ID of the subject to be used for the measurement file name."""

    logger: Logger
    """Class logger for debug, info, warning, and error messages."""

    def __init__(
        self,
        msrmt_params: list[DpoaeMsrmtParams],
        output_channels: list[int],
        mic_trans_functions: list[MicroTransferFunction] | None = None,
        out_trans_fun: OutputCalibration | None = None,
        subject: str = '',
        ear: list[str] | None = None,
        non_interactive: bool = False,
        log: Logger | None = None
    ) -> None:
        """Creates a DPOAE recorder for given measurement parameters."""
        self.logger = log or get_logger()
        self.subject = subject
        self.signals = []
        self.msrmt_info = []

        if ear is None:
            ear = ['' for _ in range(len(msrmt_params))]

        num_block_samples = 0
        num_total_recording_samples = 0
        block_duration = 0
        recording_duration = 0.0

        if len(output_channels) < 2*len(msrmt_params):
            self.logger.error(
                'Invalid number of output channels %s '
                'for number of measurements %s.',
                len(output_channels),
                len(msrmt_params)
            )
            self.msrmt = None
            return


        # Setup hardware data
        active_in_channels = list({
            b for a, b in DeviceConfig.output_input_mapping
            if a in output_channels
        })

        n_in_channels = max(
            *active_in_channels,
            DeviceConfig.sync_channels[1]
        ) + 1
        n_out_channels = max(output_channels) + 1
        hw_data = HardwareData(
            n_in_channels,
            n_out_channels,
            DeviceConfig.input_device,
            DeviceConfig.output_device,
            output_channels,
            get_input_channels(output_channels)
        )

        # Initialize signals
        self.signals = [
            ZeroSignal() for _ in range(hw_data.get_stream_output_channels())
        ]

        for i, msrmt_params_i in enumerate(msrmt_params):

            output_channels_i = hw_data.get_output_msrmt_channels(i)

            num_block_samples = int(
                msrmt_params_i['block_duration'] * DeviceConfig.sample_rate
            )
            num_total_recording_samples = (
                msrmt_params_i['num_averaging_blocks'] * num_block_samples
            )
            block_duration = num_block_samples / DeviceConfig.sample_rate
            recording_duration = (
                num_total_recording_samples / DeviceConfig.sample_rate
            )

            if mic_trans_functions:
                mic_trans_functions[i].num_samples = num_block_samples
                mic_trans_functions[i].sample_rate = DeviceConfig.sample_rate
                mic_trans_functions[i].interpolate_transfer_fun()

                mic_trans_fun_i = mic_trans_functions[i]
            else:
                mic_trans_fun_i = None

            if block_duration != msrmt_params_i['block_duration']:
                self.logger.warning(
                    'Block duration adjusted to %.2f ms',
                    block_duration * 1E3
                )

            stimulus = ContDpoaeStimulus(
                f1=0.0,
                f2=0.0,
                level1=0.0,
                level2=0.0
            )
            self.generate_output_signals(
                stimulus,
                msrmt_params_i,
                block_duration,
                num_block_samples,
                num_total_recording_samples,
                output_channels_i,
                out_calib=out_trans_fun,
            )
            msrmt_ctx = DpoaeMsrmtContext(
                fs=DeviceConfig.sample_rate,
                block_size=num_block_samples,
                non_interactive=non_interactive,
                input_trans_fun=mic_trans_fun_i,
                f1=stimulus.f1,
                f2=stimulus.f2,
                num_recorded_blocks=0
            )
            self.msrmt_info.append(
                ContDpoaeMsrmtInfo(
                    stimulus,
                    msrmt_ctx,
                    None,
                    ear[i],
                    num_total_recording_samples,
                    num_block_samples
                )
            )

        # Check for unequal sample sizes:
        if not (
            all(
            (
                obj.num_total_recording_samples
                == self.msrmt_info[0].num_total_recording_samples
            ) for obj in self.msrmt_info
            )
            and all((
                obj.num_block_samples
                == self.msrmt_info[0].num_block_samples
            ) for obj in self.msrmt_info
            )
        ):
            self.logger.error(
                'Recording duration does not match. Skipping measurement.'
            )
            self.msrmt = None
            return

        rec_data = RecordingData(
            DeviceConfig.sample_rate,
            recording_duration,
            num_total_recording_samples,
            num_block_samples,
            DeviceConfig.device_buffer_size
        )
        self.msrmt = SyncMsrmt(
            rec_data,
            hw_data,
            self.signals,
            block_duration
        )

    def record(self) -> None:
        """Starts the recording."""
        self.logger.info('Starting recording...')

        if self.msrmt is None:
            self.logger.info('Skipping measurement.')
            return

        self.msrmt.run_msrmt()

        # Plot all data and final result after user has
        # closed the live-measurement window.

        # We utilize the `ContDpoaeProcessor` to handle
        # raw data from the recorder.

        for i, msrmt_info_i in enumerate(self.msrmt_info):
            input_channel = (
                self.msrmt.hardware_data.get_unique_input_channels()[i]
            )
            recorded_signal = self.msrmt.get_recorded_signal(input_channel)
            if not recorded_signal.size:
                continue

            recording: DpoaeMsrmtData = {
                'recorded_signal': recorded_signal,
                'samplerate': DeviceConfig.sample_rate,
                'f1': msrmt_info_i.stimulus.f1,
                'f2': msrmt_info_i.stimulus.f2,
                'level1': msrmt_info_i.stimulus.level1,
                'level2': msrmt_info_i.stimulus.level2,
                'num_block_samples': msrmt_info_i.msrmt_ctx.block_size,
                'recorded_sync': self.msrmt.live_msrmt_data.sync_recorded
            }

            self.msrmt_info[i].dpoae_processor = ContDpoaeProcessor(
                recording,
                mic=msrmt_info_i.msrmt_ctx.input_trans_fun
            )

            processor = self.msrmt_info[i].dpoae_processor
            if processor is None:
                return

            processor.process_msrmt()
            if not msrmt_info_i.msrmt_ctx.non_interactive:
                self.logger.info(
                    'Showing offline results. Please close window to continue.'
                )
                processor.plot()

    def save_recording(self) -> None:
        """Stores the measurement data in binary file."""

        if self.msrmt is None:
            return

        save_path = os.path.join(
            os.getcwd(),
            'measurements'
        )
        os.makedirs(save_path, exist_ok=True)
        cur_time = datetime.now()
        time_stamp = cur_time.strftime('%y%m%d-%H%M%S')
        for i, msrmt_info_i in enumerate(self.msrmt_info):
            side_id = helpers.sanitize_filename_part(msrmt_info_i.ear)
            if len(side_id) == 0:
                output_channels_i = (
                    self.msrmt.hardware_data.get_output_msrmt_channels(i)
                )
                side_id = f'out_{output_channels_i[0]}_{output_channels_i[1]}'

            parts = [
                'cdpoae_msrmt',
                time_stamp,
                helpers.sanitize_filename_part(self.subject),
                side_id,
                str(int(msrmt_info_i.stimulus.f2)),
                str(int(msrmt_info_i.stimulus.level2)),
            ]
            file_name = '_'.join(filter(None, parts))
            file_save_path = os.path.join(save_path, file_name)
            input_channel = (
                self.msrmt.hardware_data.get_unique_input_channels()[i]
            )
            recorded_signal = self.msrmt.get_recorded_signal(input_channel)
            processor = self.msrmt_info[i].dpoae_processor
            if processor is not None:
                averaged = processor.raw_averaged
                spectrum = processor.dpoae_spectrum
            else:
                averaged = np.array(0,np.float64)
                spectrum = np.array(0,np.float64)
            np.savez(file_save_path,
                average=averaged,
                spectrum=spectrum,
                recorded_signal=recorded_signal,
                samplerate=DeviceConfig.sample_rate,
                f1=msrmt_info_i.stimulus.f1,
                f2=msrmt_info_i.stimulus.f2,
                level1=msrmt_info_i.stimulus.level1,
                level2=msrmt_info_i.stimulus.level2,
                num_block_samples = msrmt_info_i.msrmt_ctx.block_size,
                recorded_sync=self.msrmt.live_msrmt_data.sync_recorded
            )
            self.logger.info('Saved measurement to %s.npz', file_save_path)

    def generate_output_signals(
        self,
        stimulus: ContDpoaeStimulus,
        msrmt_params: DpoaeMsrmtParams,
        block_duration: float,
        num_block_samples: int,
        num_total_recording_samples: int,
        output_channels: list[int],
        out_calib: OutputCalibration | None = None,
    ) -> None:
        """Generates the output signals for playback."""
        stimulus.calculate_frequencies(
            msrmt_params,
            block_duration
        )
        stimulus.level1 = generator.calculate_pt1_level(msrmt_params)
        stimulus.level2 = msrmt_params['level2']
        stimulus1, stimulus2 = stimulus.generate_stimuli(
            num_block_samples,
            output_channels,
            output_calibration=out_calib,
        )

        # we always use rising and falling edges
        ramp_len = int(
            DeviceConfig.ramp_duration * 1E-3 * DeviceConfig.sample_rate
        )
        ramp = 0.5*(1 - np.cos(2*np.pi*np.arange(ramp_len)/(2*ramp_len)))
        ramp = ramp.astype(np.float32)

        self.signals[output_channels[0]] = PeriodicRampSignal(
            stimulus1,
            num_total_recording_samples,
            ramp
        )

        self.signals[output_channels[1]] = PeriodicRampSignal(
            stimulus2,
            num_total_recording_samples,
            ramp
        )
