Timestepping
============

:py:class:`~sympl.TimeStepper` objects use time derivatives from
:py:class:`~sympl.Prognostic` objects to step a model state forward in time.
They are initialized using a list of :py:class:`~sympl.Prognostic` objects.

.. code-block:: python

    from sympl import AdamsBashforth
    time_stepper = AdamsBashforth([MyPrognostic(), MyOtherPrognostic()])

Once initialized, a :py:class:`~sympl.TimeStepper` object has a very similar
interface to the :py:class:`~sympl.Implicit` object.

.. code-block:: python

    from datetime import timedelta
    time_stepper = AdamsBashforth([MyPrognostic()])
    timestep = timedelta(minutes=10)
    diagnostics, next_state = time_stepper(state, timestep)
    state.update(diagnostics)

The returned ``diagnostics`` dictionary contains diagnostic quantities from
the timestep of the input ``state``, while ``next_state`` is the state
dictionary for the next timestep. It is possible that some of the arrays in
``diagnostics`` may be the same arrays as were given in the input ``state``,
and that they have been modified. In other words, ``state`` may be modified by
this call. For instance, the time filtering necessary when using Leapfrog
time stepping means the current model state has to be modified by the filter.

It is only after calling the :py:class:`~sympl.TimeStepper` and getting the
diagnostics that you will have a complete state with all diagnostic quantities.
This means you will sometimes want to pass ``state`` to your
:py:class:`~sympl.Monitor` objects *after* calling
the :py:class:`~sympl.TimeStepper` and getting ``next_state``.

Keep in mind that for split-time models, multiple :py:class:`~sympl.TimeStepper`
objects might be called in in a single pass of the main loop. There are also
:py:class:`~sympl.Implicit` objects which evolve the state forward in time
without the use of Prognostic objects.

.. autoclass:: sympl.TimeStepper
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.AdamsBashforth
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.Leapfrog
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__
