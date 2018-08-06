Timestepping
============

:py:class:`~sympl.TendencyStepper` objects use time derivatives from
:py:class:`~sympl.TendencyComponent` objects to step a model state forward in time.
They are initialized using any number of :py:class:`~sympl.TendencyComponent` objects.

.. code-block:: python

    from sympl import AdamsBashforth
    time_stepper = AdamsBashforth(MyPrognostic(), MyOtherPrognostic())

Once initialized, a :py:class:`~sympl.TendencyStepper` object has a very similar
interface to the :py:class:`~sympl.Stepper` object.

.. code-block:: python

    from datetime import timedelta
    time_stepper = AdamsBashforth(MyPrognostic())
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

It is only after calling the :py:class:`~sympl.TendencyStepper` and getting the
diagnostics that you will have a complete state with all diagnostic quantities.
This means you will sometimes want to pass ``state`` to your
:py:class:`~sympl.Monitor` objects *after* calling
the :py:class:`~sympl.TendencyStepper` and getting ``next_state``.

.. warning:: :py:class:`~sympl.TendencyStepper` objects do not, and should not,
    update 'time' in the model state.

Keep in mind that for split-time models, multiple :py:class:`~sympl.TendencyStepper`
objects might be called in in a single pass of the main loop. If each one
updated ``state['time']``, the time would be moved forward more than it should.
For that reason, :py:class:`~sympl.TendencyStepper` objects do not update
``state['time']``.

There are also
:py:class:`~sympl.Stepper` objects which evolve the state forward in time
without the use of TendencyComponent objects. These function exactly the same as a
:py:class:`~sympl.TendencyStepper` once they are created, but do not accept
:py:class:`~sympl.TendencyComponent` objects when you create them. One example might
be a component that condenses all supersaturated moisture over some time period.
:py:class:`~sympl.Stepper` objects are generally used for parameterizations
that work by determining the target model state in some way, or involve
limiters, and cannot be represented as a :py:class:`~sympl.TendencyComponent`.

.. autoclass:: sympl.TendencyStepper
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
