"""Sound device display script.

This module provides a simple command-line interface for listing available
sound devices on the system. It supports the following usage patterns:

- No arguments:
    Lists all available sound devices.

- String argument:
    Filters the list of sound devices by name. Only devices whose names
    contain the given substring (case-insensitive) are shown.

- Integer argument:
    Displays detailed information about the device with the given device ID.

Typical usage example:

    python display_devices.py           # Show all devices
    python display_devices.py USB       # Show devices containing 'USB'
    python display_devices.py 3         # Show device with ID 3
"""

import sys

import sounddevice as sd


def print_info(search_param: str | int | None = None):
    """Print info about devices."""
    if search_param is None:
        search_param = ''
    elif isinstance(search_param, int):
        print(sd.query_devices(int(search_param)))
        return

    for device in sd.query_devices():
        if search_param.lower() in device['name'].lower():  # type: ignore
            print(f'{device}\n')


def run_cli():
    """Run printing info with console arguments."""
    # Obtain argument if given an possibly convert to integer
    if len(sys.argv) > 1:
        search_string = sys.argv[1]
        if search_string.isdigit():
            print_info(int(search_string))
        else:
            print_info(search_string)
    else:
        print_info()


if __name__ == '__main__':
    # Obtain argument if given an possibly convert to integer
    run_cli()
