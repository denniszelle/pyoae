"""Default logging module for PyOAE recorders.

This module provides a simplified interface to retrieve
a color-coded logger for the terminal.
"""

import logging


class ColorFormatter(logging.Formatter):
    """Class defining the color coding of the logger."""
    COLORS = {
        logging.DEBUG: "\033[37m",   # white/gray
        logging.INFO: "\033[36m",    # cyan
        logging.WARNING: "\033[33m", # yellow/orange
        logging.ERROR: "\033[31m",   # red
        logging.CRITICAL: "\033[41m",# white on red bg
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging(level=logging.INFO) -> None:
    """Sets up the logger format."""
    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(ColorFormatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_pyoae_logger(name: str = 'PyOAE') -> logging.Logger:
    """Sets up and returns a logger instance for PyOAE recording scripts."""
    setup_logging()
    return logging.getLogger(name)
