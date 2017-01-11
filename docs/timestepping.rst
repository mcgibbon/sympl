Timestepping
============

:py:class:`sympl.TimeStepper` objects use time derivatives from
:py:class:`sympl.Prognostic` objects to step a model state forward in time.

Initialization
--------------

:py:class:`sympl.TimeStepper` objects are initialized using a list of
:py:class:`sympl.Prognostic` objects.

.. code-block:: python

    from sympl import AdamsBashforth
    time_stepper = AdamsBashforth([MyPrognostic(), MyOtherPrognostic()])

Usage
-----

Once initialized, a :py:class:`sympl.TimeStepper` object has a very similar
interface to the :py:class:`Implicit` object.

.. code-block:: python

    from datetime import timedelta
    time_stepper = AdamsBashforth([MyPrognostic()])
    timestep = timedelta(minutes=10)
    diagnostics, next_state = time_stepper(state, timestep)
    state.update(diagnostics)

The returned ``diagnostics`` dictionary contains diagnostic quantities from
the timestep of the input ``state``, while ``next_state`` is the state
dictionary for the next timestep.

It is only after calling the :py:class:`sympl.TimeStepper` and getting the
diagnostics that you will have a complete state with all diagnostic quantities.
This means you will sometimes want to pass ``state`` to your
:py:class:`sympl.Monitor` objects *after* calling
the :py:class:`sympl.TimeStepper` and getting ``next_state``.

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
