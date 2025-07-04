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

* Python ≥ 3.10

### Dependencies

| Package       | Version (tested) |
| ------------- | ---------------- |
| `sounddevice` | 0.5.1            |
| `numpy`       | 2.2.5            |
| `scipy`       | 1.15.2           |
| `matplotlib`  | 3.10.1           |

---

## Tutorials and Hardware

We provide examples, instructions, and tutorials on the [EARLAB website](https://www.earlab.de/en-insights) demonstrating how to connect EARLAB research amplifiers and OAE probes to commercial sound cards for OAE recordings.

## Getting Started

### 1. List Available Devices

Use the helper script to list connected audio devices and identify your sound card.
Make sure to run the command from the project root directory (the parent folder containing both the `examples` and `pyoae` directories):

```bash
python3 -m examples.display_devices
```

You can also filter the output by device name:

```bash
python3 -m examples.display_devices MAYA
```

### 2. Configure the Audio Device

Based on the results from the `display_devices` script, configure the audio interface to be used for recordings in the `examples.device_config` module.

For example, to select a sound card identified by the string 'USB' and set the sampling rate to 48 kHz, update the following constants in `device_config`:

```python
INPUT_DEVICE: int | str = 'USB'
OUTPUT_DEVICE: int | str = 'USB'
FS: float = 48000.
```

You can also adjust the audio device’s buffer size using:

```python
DEVICE_BUFFER_SIZE: int = 4096
```

If you experience dropouts during playback or recording, increasing the buffer size often resolves the issue. A larger buffer is typically required for higher sampling rates (e.g., 192 kHz).

### 3. Run an Example Recording

#### Spontaneous Otoacoustic Emissions (SOAEs)

To run a basic recording without stimulus output, execute the `record_soae` script from the project root directory:

```bash
python3 -m examples.record_soae
```

#### Distortion-Product Otoacoustic Emissions (DPOAEs)

To stimulate the ear using two continuous primary tones and record a DPOAE, run the `record_dpoae` script:

```bash
python3 -m examples.record_dpoae
```

---

## Things to Consider

* **Sound Card Drivers**: PyOAE typically uses the sound card drivers provided by the operating system. If you encounter issues detecting your device or setting the sampling rate, we recommend installing the official drivers from the device manufacturer.
* **Audio Configuration**: Ensure the correct input and output device identifiers are defined in the `device_config` module. Note that device indices may change when USB audio interfaces are connected or disconnected, so double-check your configuration if problems occur.

---

## Documentation

We provide a comprehensive online documentation of the API on the [EARLAB website](https://www.earlab.de/doc/pyoae/).

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
