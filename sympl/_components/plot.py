from .._core.base_components import Monitor
from .._core.exceptions import DependencyException
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is None:
    # If dependency is not installed, use a dummy object that will alert the
    # user they need to install the dependency if they try to use it
    class PlotFunctionMonitor(Monitor):

        def __init__(self, filename):
            raise DependencyException(
                'matplotlib must be installed to use PlotFunctionMonitor')

        def store(self, state):
            pass

else:
    class PlotFunctionMonitor(Monitor):
        """
        A Monitor which uses a user-defined function to draw figures using model
        state.
        """

        def __init__(self, plot_function, interactive=True):
            """
            Initialize a PlotFunctionMonitor.

            Args:
                plot_function (func): A function plot_function(fig, state) that
                    draws the given state onto the given (initially clear) figure.
                interactive (bool, optional): If true, matplotlib's interactive
                    mode will be enabled, allowing plot animation while other
                    computation is running.
            """
            if interactive:
                plt.ion()
            self._plot_function = plot_function
            self._fig = plt.figure()

        def store(self, state):
            """
            Updates the plot using the given state.

            Args:
                state (dict): A model state dictionary.
            """
            self._fig.clear()
            self._plot_function(self._fig, state)
            plt.draw_all()
            plt.pause(1e-5)  # necessary to draw, pause can be arbitrarily small
