========
Monitors
========

:py:class:`~sympl.Monitor` objects store states in some way, whether it is by
displaying the new state on a plot that is shown to the user, updating
information on a web server, or saving the state to a file. They are called
like so:

.. code-block:: python

    monitor = MyMonitor()
    monitor.store(state)

The :py:class:`~sympl.Monitor` will take advantage of the 'time' key in the
``state`` dictionary in order to determine the model time of the state. This is
particularly important for a :py:class:`~sympl.Monitor` which outputs a series
of states to disk.

.. autoclass:: sympl.Monitor
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.NetCDFMonitor
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.PlotFunctionMonitor
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__
