===============
Component Types
===============

In Sympl, computation is mainly performed using :py:class:`sympl.Prognostic`,
:py:class:`sympl.Diagnostic`, and :py:class:`sympl.Implicit` objects.
Each of these types, once initialized, can be passed in a current model state.
:py:class:`sympl.Prognostic` objects use the state to return tendencies and
diagnostics at the current time. :py:class:`sympl.Diagnostic` objects
return only diagnostics from the current time. :py:class:`sympl.Implicit`
objects will take in a timestep along with the state, and then return the
next state as well as modifying the current state to include more diagnostics
(it is similar to a :py:class:`sympl.TimeStepper` in how it is called).

These classes themselves (listed in the previous paragraph) are not ones you
can initialize (e.g. there is no one 'prognostic' scheme), but instead should
be subclassed to contain computational code relevant to the model you're
running.

Prognostic
----------

As stated above, :py:class:`sympl.Prognostic` objects use the state to return
tendencies and diagnostics at the current time. In a full model, the tendencies
are used by a time stepping scheme (in Sympl, a :py:class:`sympl.TimeStepper`)
to determine the values of quantities at the next time.

You can call a :py:class:`sympl.Prognostic` directly to get diagnostics and
tendencies like so:

.. code-block:: python

    radiation = RRTMRadiation()
    diagnostics, tendencies = radiation(state)

``diagnostics`` and ``tendencies`` in this case will both be dictionaries,
similar to ``state``. Even if the :py:class:`sympl.Prognostic` being called
does not compute any diagnostics, it will still return an empty
diagnostics dictionary.

Usually, you will call a Prognostic object through a
:py:class:`sympl.TimeStepper` that uses it to determine values at the next
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

:py:class:`sympl.Diagnostic` objects use the state to return quantities
('diagnostics') from the same timestep as the input state. You can call a
:py:class:`sympl.Diagnostic` directly to get diagnostic quantities like so:

.. code-block:: python

    diagnostic_component = MyDiagnostic()
    diagnostics = diagnostic_component(state)

Instead of returning a new dictionary with the additional diagnostic quantities,
a :py:class:``Diagnostic`` can update the state dictionary in-place with the new
quantities. You do this like so:

.. code-block:: python

    diagnostic_component = MyDiagnostic()
    diagnostic_component.update_state(state)

The ``update_state`` call has the advantage that it will automatically check to
see if it is overwriting any quantities already present in state, and will
raise a :py:class:`sympl.SharedKeyException` before doing so. This ensures you
don't have multiple pieces of code trying to output the same diagnostic, with
one overwriting the other.

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

:py:class:`sympl.Implicit` objects use a state and a timestep to return the next
state, and update the input state with any relevant diagnostic quantities. You
can call an Implicit object like so:

.. code-block:: python

    from datetime import timedelta
    implicit = MyImplicit()
    timestep = timedelta(minutes=10)
    next_state = implicit(state, timestep)

Following the ``implicit`` call, ``state`` will have been modified in-place to
include any diagnostics produced by the :py:class:`sympl.Implicit` component
for the timestep of the input state.

This is important, so we'll repeat it:
**the input state can be modified by the call to the ``Implicit`` object**.

.. autoclass:: sympl.Implicit
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__
