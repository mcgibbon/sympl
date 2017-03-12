===============
Component Types
===============

In Sympl, computation is mainly performed using :py:class:`~sympl.Prognostic`,
:py:class:`~sympl.Diagnostic`, and :py:class:`~sympl.Implicit` objects.
Each of these types, once initialized, can be passed in a current model state.
:py:class:`~sympl.Prognostic` objects use the state to return tendencies and
diagnostics at the current time. :py:class:`~sympl.Diagnostic` objects
return only diagnostics from the current time. :py:class:`~sympl.Implicit`
objects will take in a timestep along with the state, and then return the
next state as well as modifying the current state to include more diagnostics
(it is similar to a :py:class:`~sympl.TimeStepper` in how it is called).

These classes themselves (listed in the previous paragraph) are not ones you
can initialize (e.g. there is no one 'prognostic' scheme), but instead should
be subclassed to contain computational code relevant to the model you're
running.

Prognostic
----------

As stated above, :py:class:`~sympl.Prognostic` objects use the state to return
tendencies and diagnostics at the current time. In a full model, the tendencies
are used by a time stepping scheme (in Sympl, a :py:class:`~sympl.TimeStepper`)
to determine the values of quantities at the next time.

You can call a :py:class:`~sympl.Prognostic` directly to get diagnostics and
tendencies like so:

.. code-block:: python

    radiation = RRTMRadiation()
    diagnostics, tendencies = radiation(state)

``diagnostics`` and ``tendencies`` in this case will both be dictionaries,
similar to ``state``. Even if the :py:class:`~sympl.Prognostic` being called
does not compute any diagnostics, it will still return an empty
diagnostics dictionary.

Usually, you will call a Prognostic object through a
:py:class:`~sympl.TimeStepper` that uses it to determine values at the next
timestep.

.. autoclass:: sympl.Prognostic
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.ConstantPrognostic
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.RelaxationPrognostic
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

Diagnostic
----------

:py:class:`~sympl.Diagnostic` objects use the state to return quantities
('diagnostics') from the same timestep as the input state. You can call a
:py:class:`~sympl.Diagnostic` directly to get diagnostic quantities like so:

.. code-block:: python

    diagnostic_component = MyDiagnostic()
    diagnostics = diagnostic_component(state)

You should be careful to check in the documentation of the particular
:py:class:`~sympl.Diagnostic` you are using to see whether it modifies the
``state`` given to it as input. :py:class:`~sympl.Diagnostic` objects in charge
of updating ghost cells in particular may modify the arrays in the input
dictionary, so that the arrays in the returned ``diagnostics`` dictionary are
the same ones as were sent as input in the ``state``. To make it clear that
the state is being modified when using such objects, we recommend using a
syntax like:

.. code-block:: python

    state.update(diagnostic_component(state))

.. autoclass:: sympl.Diagnostic
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.ConstantDiagnostic
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

Implicit
--------

:py:class:`~sympl.Implicit` objects use a state and a timestep to return the next
state, and update the input state with any relevant diagnostic quantities. You
can call an Implicit object like so:

.. code-block:: python

    from datetime import timedelta
    implicit = MyImplicit()
    timestep = timedelta(minutes=10)
    diagnostics, next_state = implicit(state, timestep)
    state.update(diagnostics)

The returned ``diagnostics`` dictionary contains diagnostic quantities from
the timestep of the input ``state``, while ``next_state`` is the state
dictionary for the next timestep. It is possible that some of the arrays in
``diagnostics`` may be the same arrays as were given in the input ``state``,
and that they have been modified. In other words, ``state`` may be modified by
this call. For instance, the object may need to update ghost cells in the
current state. Or if an object provides 'cloud_fraction' as a diagnostic, it
may modify an existing 'cloud_fraction' array in the input state if one is
present, instead of allocating a new array.

.. autoclass:: sympl.Implicit
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__
