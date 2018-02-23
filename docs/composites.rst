==========
Composites
==========

There are a set of objects in Sympl that wrap multiple components into a single
object so they can be called as if they were one component. There is one each
for :py:class:`~sympl.Prognostic`, :py:class:`~sympl.Diagnostic`, and
:py:class:`~sympl.Monitor`. These can be used to simplify code, so that
the way you call a list of components is the same as the way you would
call a single component. For example, *instead* of writing:

.. code-block:: python

    prognostic_list = [
        MyPrognostic(),
        MyOtherPrognostic(),
        YetAnotherPrognostic(),
    ]
    all_diagnostics = {}
    total_tendencies = {}
    for prognostic_component in prognostic_list:
        tendencies, diagnostics = prognostic_component(state)
        # this should actually check to make sure nothing is overwritten,
        # but this code does not
        total_tendencies.update(tendencies)
        for name, value in tendencies.keys():
            if name not in total_tendencies:
                total_tendencies[name] = value
            else:
                total_tendencies[name] += value
        for name, value in diagnostics.items():
            all_diagnostics[name] = value

You could write:

.. code-block:: python

    prognostic_composite = PrognosticComposite([
        MyPrognostic(),
        MyOtherPrognostic(),
        YetAnotherPrognostic(),
    ])
    tendencies, diagnostics = prognostic_composite(state)

This second call is much cleaner. It will also automatically detect whether
multiple components are trying to write out the same diagnostic, and raise
an exception if that is the case (so no results are being silently
overwritten). You can get similar simplifications for
:py:class:`~sympl.Diagnostic` and :py:class:`~sympl.Monitor`.

.. note:: PrognosticComposites are mainly useful inside of TimeSteppers, so
          if you're only writing a model script it's unlikely you'll need them.

API Reference
-------------

.. autoclass:: sympl.PrognosticComposite
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.DiagnosticComposite
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. autoclass:: sympl.MonitorComposite
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__
