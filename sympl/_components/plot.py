from .._core.base_components import Monitor
from .._core.exceptions import DependencyError
from .._core.array import DataArray


def copy_state(state):
    return_state = {}
    for name, quantity in state.items():
        if isinstance(quantity, DataArray):
            return_state[name] = DataArray(
                quantity.values.copy(), quantity.coords, quantity.dims,
                quantity.name, quantity.attrs)
        else:
            return_state[name] = quantity
    return return_state


class PlotFunctionMonitor(Monitor):
    """
    A Monitor which uses a user-defined function to draw figures using model
    state.
    """

    def __init__(self, plot_function, interactive=True):
        """
        Initialize a PlotFunctionMonitor.

        Args
        ----
        plot_function : func
            A function plot_function(fig, state) that
            draws the given state onto the given (initially clear) figure.
        interactive: bool, optional
            If true, matplotlib's interactive mode will be enabled,
            allowing plot animation while other computation is running.
        """
        global plt
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise DependencyError(
                'matplotlib must be installed to use PlotFunctionMonitor')
        if interactive:
            plt.ion()
            self._fig = plt.figure()
        else:
            plt.ioff()
            self._fig = None
        self._plot_function = plot_function

    @property
    def interactive(self):
        return self._fig is not None

    def store(self, state):
        """
        Updates the plot using the given state.

        Args
        ----
        state : dict
            A model state dictionary.
        """
        if self.interactive:
            self._fig.clear()
            fig = self._fig
        else:
            fig = plt.figure()

        self._plot_function(fig, copy_state(state))

        fig.canvas.draw()
        if not self.interactive:
            plt.show()
