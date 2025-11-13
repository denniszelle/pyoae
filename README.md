# PyOAE

**PyOAE** is an open-source Python framework designed for basic otoacoustic emission (OAE) measurements, including **Spontaneous Otoacoustic Emissions (SOAEs)** and **Distortion-Product Otoacoustic Emissions (DPOAEs)**.

The goal of this project is to offer a lightweight, accessible solution for conducting OAE measurements using consumer-grade sound cards as affordable and widely available acquisition hardware.

Designed with simplicity and flexibility in mind, PyOAE is suitable for quick experimental use as well as for extending and integrating custom measurement and analysis tools.

The software is actively developed, and future updates may include features such as automated calibration routines.

---

## Requirements

To use **PyOAE**, we recommend installing the dependencies in a virtual environment to avoid conflicts with other Python packages.

### Install via pip

```bash
python3 -m venv .venv  # or python -m venv .venv (depending on your Python installation)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Python Version

We recommend the use of **Python 3.12** or higher.

### Dependencies

The following package versions work without known issues on Windows 10 using Python 3.14. We do not expect any issues with earlier Python versions of down to 3.12.


| Package               | Version (tested) | Notes                        |
| --------------------- | ---------------- | ----------------------------- |
| `sounddevice`         | 0.5.3            |                               |
| `numpy`               | 2.3.4            |                               |
| `scipy`               | 1.16.2           |                               |
| `matplotlib`          | 3.10.7           |                               |
| `typing_extensions`   | 4.15.0            | Only required for Python < 3.11 |

---

## Tutorials and Hardware

We provide examples, instructions, and tutorials on the [EARLAB website](https://www.earlab.de/en-insights) demonstrating how to connect EARLAB research amplifiers and OAE probes to commercial sound cards for OAE recordings.

## Getting Started

### 1. List Available Devices

Use the helper script to list connected audio devices and identify your sound card.
Make sure to run the command from the project root directory (the parent folder containing both the recording scripts and the `pyoae` package) with an active virtual environment:

```bash
python3 -m pyoae.device.display_devices
```

You can also filter the output by device name (*highly recommended on Windows*):

```bash
python3 -m pyoae.device.display_devices Focusrite
```

The output lists key properties such as the device name, number of input and output channels, and the sample rate. For devices that support multiple sample rates and bit depths, select the appropriate option in your audio system settings, for example, in Audio MIDI Setup (macOS), the operating system’s audio settings, or the control software for your audio interface (e.g., Focusrite Control).

**NOTE**: The `display_devices` script often lists a large number of available devices because Windows treats each combination of channel count and sampling frequency as a separate device instance. In addition, even if only a single audio interface is connected, Windows typically shows separate instances for input and output channels, which need to be selected individually (see the next step below).

### 2. Configure the Audio Device

Based on the results from the `display_devices` script, configure the audio interface to be used for recordings in the `device_config.json` file.

For example, to select a Focusrite Scarlett 2i2 audio interface and set the sampling rate to 96 kHz on macOS, update the following attributes in `device_config.json`:

```json
    "input_device": "Scarlett 2i2 USB",
    "output_device": "Scarlett 2i2 USB",
    "sample_rate": 96000.0,
    "device_buffer_size": 4096,
```

On Windows, we recommend using the device index instead of the device name. To select a Focusrite Scarlett 2i2 audio interface with a sampling rate of 48 kHz, use the following configuration:

```json
    "input_device": 14,
    "output_device": 12,
    "sample_rate": 48000.0,
    "device_buffer_size": 4096,
```

*Please note that the device names and indices may differ depending on your operating system and specific hardware.*

For a sample rate of 96 kHz, we recommend using a buffer size for the audio device of 4096 frames. If you experience dropouts during playback or recording, increasing the buffer size often resolves the issue. A larger buffer is typically required for higher sampling rates (e.g., 8192 frames for 192 kHz).

### 3. Perform Basic System Calibration

#### Input or Microphone Calibration

PyOAE can convert recorded audio signals into sound pressure levels using a microphone calibration loaded from a JSON configuration file. The file `mic_calib.json` in the templates folder demonstrates the required format. The calibration file includes a `"transfer_function"` section that defines the amplitude and phase characteristics of the microphone. In the template file, this transfer function has unity gain and zero phase delay.

The `"abs_calibration"` section in the calibration file defines the sensitivity of the complete input channel, which can be obtained using the absolute calibration script `calibrate_sensitivity.py`. Because sensitivity depends on both the microphone and the audio interface (including its input gain setting), it is strongly recommended to perform an absolute calibration for your own setup.

To perform an absolute calibration with PyOAE, use a reference calibrator (typically generating a 1000 Hz tone at 94 dB SPL). Insert the microphone probe into the calibrator port and ensure a proper seal. For EARLAB R30e in-ear probes, use a suitable ear tip when inserting the probe tip into the calibrator cavity.

In the terminal, run the following command to start the absolute calibration of the input channel (with the reference calibrator active):

```bash
python -m calibrate_sensitivity
```

The script displays messages about the measurement settings and progress in the terminal. After the recording, the calibration result (assuming a reference tone of 1000 Hz at 94 dB SPL) is displayed in the terminal:

```text
[INFO] pyoae: Recording complete. Please close window to continue.
[INFO] pyoae: Closing stream.
[INFO] pyoae: Showing offline results. Please close window to continue.
[INFO] pyoae: Reference input: -12.69 dBFS -> Input pressure at full scale:  6049419.05.
```

The example calibration result above shows that, for the current setup, a peak amplitude of 6,049,419.05 µPa corresponds to an amplitude of 1.0 relative to full scale. Copy this value into the `"sensitivity"` property in the `"abs_calibration"` section of your microphone calibration file. Adjusting the input gain to achieve a reference input between -20.0 and -10 dBFS is recommended.

We recommend storing your custom microphone calibration files in the `mic` folder at the project’s root directory. This folder is excluded from version control via .gitignore and might need to be created manually.

#### Speaker Calibration

PyOAE offers a very simple output calibration via the script `record_output_calib.py`. For each output channel, a multi tone is presented to characterize the output-channel response (typically dominated by the speaker, the ear-canal anatomy, and the ear-probe fit). In the terminal, run the following command to start the output calibration with the ear probe inserted into the ear canal of the measurement subject (make sure to replace the microphone-calibration file with that matching your own setup):

```bash
python3 -m record_output_calib --mic 'mic/mic.json' --save
```

During the measurement, the recorded signal is shown in a figure window. After closing it, the script displays the recorded amplitude response for both channels. The calibration results are stored in the measurement folder in the project's root directory as a JSON file with a unique time stamp, as indicated by the INFO message in the terminal:

```text
[INFO] pyoae.pyoae.calibrator: Recording complete. Please close window to continue.
[INFO] pyoae: Closing stream.
[INFO] pyoae: Showing offline results. Please close window to continue.
[INFO] pyoae.pyoae.files: Output calibration saved to 250919-100413_out_calib.json.
[INFO] PyOAE Output Calibrator: Calibration saved with time stamp: 250919-100413.
```

**Please note that PyOAE currently offers only a simplified calibration of the amplitude response. Support to calibrate for phase characteristics will be added in future revisions of the library.**

### 4. Run a Recording

#### Measurement Protocols

You can create your own measurement sequences using JSON protocol files. The `templates` folder comprises a few examples to record conventional DPOAEs with continuous stimulation as well as DPOAE responses in the time domain using pulsed and/or short-pulsed stimulation.

The protocols contain a list of `"msrmts"` with each block defining the properties of the recording, such as the length of a single acquisition block in seconds, the number of blocks used for averaging and the stimulus frequencies and levels. In the following example with just one measurement, the frequency f1 and its stimulus level L1 will be calculated by PyOAE.

```json
{
    "msrmts": [
        {
            "block_duration": 0.1,
            "num_averaging_blocks": 100,
            "f2": 2000,
            "f1": null,
            "f2f1_ratio": 1.2,
            "level2": 45,
            "level1": null
        }
    ]
}
```

Here, f1 is calculated using the commonly used frequency ratio, f2/f1 = 1.2. L1 is calculated using the scissor-paradigm by [Kummer et al. (1998)](https://doi.org/10.1121/1.423054), L1 = 0.4 * L2 + 39 dB. Specified values for `f1` and `level1` will take precedence over these automatically calculated values.

For pulsed stimulation, the onset, offset, and shape of the pulses can be specified using the time markers (in ms) in each block in the `"msrmt"` list of the protocol, as shown in the following example:

```json
    "f1_pulse": {
        "t_rise": 5.0,
        "t_fall": 5.0,
        "t_on": 1.0,
        "duration": 78.0,
        "is_short_pulse": false
    },
    "f2_pulse": {
        "t_rise": 2.5,
        "t_fall": 2.5,
        "t_on": 10.0,
        "duration": 9.0,
        "is_short_pulse": true
    }
```

For more information on pulsed DPOAEs, see [Zelle et al. (2017)](https://link.springer.com/article/10.1007/s00106-016-0267-y).

**Please note that PyOAE currently requires that the windows with the measurement results be closed before the next measurement in the list is started.**

#### Spontaneous Otoacoustic Emissions (SOAEs)

To run a basic recording without stimulus output, execute the `record_soae` script from the project root directory:

```bash
python3 -m record_soae
```

To save the recording, add the argument `--save`:

```bash
python3 -m record_soae --save
```

If no protocol for the SOAE recording is specified, the script will use the default parameters as stated in the terminal:

```text
[ERROR] pyoae.pyoae.files: SOAE protocol  not found.
[WARNING] pyoae.pyoae.files: Using default SOAE protocol instead.
[INFO] pyoae: Starting SOAE recording...
[INFO] pyoae: Beginning to stream.
[INFO] pyoae: Measured latency: 132.2708 ms (12698 samples).
[INFO] pyoae: 45056 samples acquired for latency determination.
[INFO] pyoae: Recording complete. Please close window to continue.
[INFO] pyoae: Closing stream.
[INFO] pyoae: Showing offline results. Please close window to continue.
```

**Please note that the SOAE recorder currently does not support an input calibration file to convert the recorded spectrum to dB SPL.**

An SOAE recording made in an artificial ear or cavity can also help detect disturbances in the measurement setup, such as power-line interference that appears at multiples of 50 or 60 Hz.

#### Distortion-Product Otoacoustic Emissions (DPOAEs)

To stimulate the ear using two continuous primary tones and record a DPOAE, run the `record_dpoae` script. To run a protocol example given in the `templates` folder, use the following command in the terminal. Ensure that you alter the example to match your microphone file and the time stamp of the previously performed output calibration.

```bash
python3 -m record_dpoae --mic 'mic/mic.json' --protocol 'templates/tpl_cdpoae.json' --calib '250919-100413'
```

Using the command line arguments `--save`, `--subject`, and `--ear`, you can save the measurement results using a subject identifier and a text describing the recording side. For example:

```bash
python3 -m record_dpoae --mic 'mic/mic.json' --protocol 'templates/tpl_cdpoae.json' --calib '250919-100413' --subject 'S000' --ear 'right' --save
```

**Please note that PyOAE currently does not correct for the input and output channel phase characteristics.**

#### Pulsed Distortion-Product Otoacoustic Emissions (pDPOAEs)

To stimulate the ear using two pulsed primary tones and record a pulsed DPOAE response in the time domain, run the `record_pulse_dpoae` script (with similar options as for continuous acquisition):

```bash
python3 -m record_pulse_dpoae --mic 'mic/mic.json' --protocol 'templates/tpl_pdpoae.json' --calib '250919-100413' --subject 'S000' --ear 'right'  --save
```

In order to obtain a time-domain signal of the DPOAE response, PyOAE utilized Primary-Tone Phase Variation ([Whitehead et al. (1996)](https://doi.org/10.1121/1.416065)). Suitable phase shifts of the primary tones enable their cancellation during the averaging process while maintaining the distortion-product at the cubic difference frequency fdp = 2f1-f2.

---

## Things to Consider

* **Sound Card Drivers**: PyOAE typically uses the sound card drivers provided by the operating system. If you encounter issues detecting your device or setting the sampling rate, we recommend installing the official drivers from the device manufacturer.
* **Audio Configuration**: Ensure the correct input and output device identifiers are defined in the `device_config` module. Note that device indices may change when USB audio interfaces are connected or disconnected, so double-check your configuration if problems occur.
* **System Sounds**: It is recommended to disable system sound notifications in your operating system. Additionally, set your computer’s default playback device (e.g., built-in speakers) for all other applications. Avoid using the audio interface assigned to PyOAE as the system’s default output device to prevent unwanted sounds from being played through the probe and interfering with the measurements.

---

## Troubleshooting

The following points may help resolve common issues encountered during measurement.

### Inconsistent Audio Playback or Recording

If you experience dropouts during playback or recording, consider the following::

* **Background Processes**: Close unnecessary applications and background services (e.g., cloud sync tools) while running measurements.
* **System Notifications**: Disable or limit operating system notifications to prevent interruptions.
* **Power Settings**: Ensure your computer is running in high-performance mode and that any battery-saving features are disabled to avoid CPU throttling.
* **Python Version**: Update to the latest stable Python release. Performance improvements are especially noticeable when upgrading from Python 3.10 or older.

---

## Documentation

For detailed information on the code, please refer to the [API online documentation](https://www.earlab.de/doc/pyoae/).

## Support

We welcome feedback, questions, and contributions!

* Report issues via the [GitHub Issue Tracker](https://github.com/denniszelle/pyoae/issues)
* Join the conversation on [GitHub Discussions](https://github.com/denniszelle/pyoae/discussions)
* Contact us via [info@earlab.de](mailto:info@earlab.de)

## Acknowledgements

This project relies heavily on the excellent [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) package. For more information, see [our acknowledgements](acknowledgements.md).

Low-level audio functionality is enabled through:

* [PortAudio](https://www.portaudio.com/)
* [ALSA Project](https://www.alsa-project.org/) (for audio support on Linux)
