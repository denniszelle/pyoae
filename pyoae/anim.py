"""Animation helpers for visualization."""

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import TimerBase
from matplotlib.figure import Figure


class MsrmtFuncAnimation(FuncAnimation):
    """Animation class to visualize a PyOAE measurement."""

    t: TimerBase | None
    """Timer to evoke single-shot callback closing the figure."""

    _msrmt_fig: Figure
    """Handle to figure with online measurement results."""

    done: bool

    def __init__(self, fig: Figure,  *args, **kwargs) -> None:
        self.t = None
        self._msrmt_fig = fig
        self.done = False
        super().__init__(fig, *args, **kwargs)

    def stop_animation(self) -> None:
        """Stops the animation and closes the figure.

        This method is called to prevent the user from closing the figure
        with measurement data in order to continue with the next measurement
        from the recording protocol.
        """
        try:
            if self.event_source is not None:
                self.event_source.stop()
        finally:
            self.t = self._msrmt_fig.canvas.new_timer(interval=0)
            self.t.single_shot = True
            self.t.add_callback(self.close_fig)
            self.t.start()

    def close_fig(self) -> None:
        """Closes the figure associated with the animation."""
        plt.close(self._msrmt_fig)
        self.done = True
