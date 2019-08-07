=======
Tracers
=======

In an Earth system model, "tracer" refers to quantities that are passively
moved around in a model without actively interacting with a component. Generally
these are moved around by a dynamical core or subgrid advection scheme. It is
possible for components to do something else to tracers (let us know if you
think of something!) but for now let's assume that's what's going on.

If a component moves around tracers, it will have its ``uses_tracers`` property
set to ``True``, and will also have a ``tracer_dims`` property set.

You can tell Sympl that you want components to move a tracer around by
registering it with :py:func:`~sympl.register_tracer`.

.. autofunction:: sympl.register_tracer

To see the current list of registered tracers, you can call
:py:func:`~sympl.get_tracer_names` or
:py:func:`~sympl.get_tracer_unit_dict`.

.. autofunction:: sympl.get_tracer_names
.. autofunction:: sympl.get_tracer_unit_dict
