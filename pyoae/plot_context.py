"""Classes providing the context for a data plots.

This module contains classes that provide references to
plot and window instances.
"""

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


@dataclass
class PlotContext:
    """Container with parameters and instances for measurement plots."""

    fig: Figure
    """Figure corresponding to the plot window."""

    axes: Axes
    """Axis object of the time plot."""

    line: Line2D
    """Line object for the time plot."""

    update_interval: float
    """Interval to apply processing and plot update during measurement."""

    live_display_duration: float
    """Duration to display time domain plot in ms."""


@dataclass
class SpectralPlotContext(PlotContext):
    """Context of plots for SOAE and continuous DPOAE recordings."""

    axes_spec: Axes
    """Axis object of the spectral plot."""

    line_spec: Line2D
    """Line object for the spectral plot."""
