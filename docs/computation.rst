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

In specific cases, it may be necessary to use a :py:class:`~sympl.ImplicitPrognostic`
object, which is discussed at the end of this section.

These classes themselves (listed in the previous paragraph) are not ones you
can initialize (e.g. there is no one 'prognostic' scheme), but instead should
be subclassed to contain computational code relevant to the model you're
running.

In addition to the computational functionality below, all components have "properties"
for their inputs and outputs, which are described in the section
:ref:`Input/Output Properties`.

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

Input/Output Properties
-----------------------

You may have noticed when reading the documentation for the classes above that
there are a number of attributes with names like ``input_properties`` for
components. These attributes give a fairly complete description of the inputs
and outputs of the component.

You can access them like this (for an example :py:class:`~sympl.Prognostic`
class ``RRTMRadiation``):

.. code-block:: python

    radiation = RRTMRadiation()
    radiation.input_properties
    radiation.diagnostic_properties
    radiation.tendency_properties

Input
*****

All components have input_properties, because they all take inputs. This
attribute (like all the other properties attributes) is a python ``dict``,
or "dictionary" (if you are unfamiliar with these, please read the `Python
documentation for dicts`_).

An example input_properties would be

.. code-block:: python

    {
        'air_temperature': {
            'dims': ['*', 'z'],
            'units': 'degK',
        },
        'vertical_wind': {
            'dims': ['*', 'z'],
            'units': 'm/s',
            'match_dims_like': ['air_temperature']
        }
    }

Each entry in the input_properties dictionary is a quantity that the object
requires as an input, and its value is another dictionary that tells you how
the object uses that quantity. The ``units`` property is the units used
internally in the object. You don't need to pass in the quantity with those
those units, as long as the units can be converted, but if you do use the same
units in the input state it will avoid the computational cost of
converting units.

The ``dims`` property can be more confusing, but is very useful. It says what
dimensions the component uses internally for those quantities. The component
requires that you give it quantities that can be transformed into those
internal dimensions, but it can take care of that transformation itself. In
this example, it will transform the arrays for both quantities to put the
vertical dimension last, and collect all the other dimensions into a single
first dimension. If you pass this object arrays that have their vertical
dimension last, it may speed up the computation, depending on the component
(but not for all components!).

So what are '*' and 'z' anyways? These are *wildcard* dimensions. 'z' will
match any dimension that is vertical, while '*' will match *any* dimension that
is not specified somewhere else in the ``dims`` list. There are also 'x' and
'y' for horizontal dimensions. The directional matches are given to Sympl
using the functions :py:func:`~sympl.set_direction_names` or
:py:func:`~sympl.add_direction_names`. If you're using someone else's package
for a component, it is likely that they call these functions for you, so you
don't have to (and if you're writing such a package, you should use
:py:func:`~sympl.add_direction_names`).

If a component is using a wildcard it
means it doesn't care very much about those directions. For example, a column
component like radiation will simply call itself on each column of the domain,
so it doesn't care about the specifics of what the non-vertical dimensions are,
as long as the desired quantities are co-located.

That's where ``match_dims_like`` comes in. This property says the object
requires all shared wildcard dimensions between the two quantity match the
same dimensions as the other
specified quantity. In this case, it will ensure that ``vertical_wind`` is on
the same grid as ``air_temperature``.

Let's consider a slight variation on the earlier example:

.. code-block:: python

    {
        'air_temperature': {
            'dims': ['*', 'mid_levels'],
            'units': 'degK',
        },
        'vertical_wind': {
            'dims': ['*', 'interface_levels'],
            'units': 'm/s',
            'match_dims_like': ['air_temperature']
        }
    }

This version requires that ``air_temperature`` be on the ``mid_levels`` vertical
grid, while ``vertical_wind`` is on the ``interface_levels``. It still requires
that all other dimensions are the same between the two quantities, so that they
are on the same horizontal grid (if they have a horizontal grid).

Outputs
*******

There are a few output property dictionaries in Sympl: ``tendency_properties``,
``diagnostic_properties``, and ``output_properties``. They are all formatted
the same way with the same properties, but tell you about the tendencies,
diagnostics, or next state values that are output by the component,
respectively.

Here's an example output dictionary:

.. code-block:: python

    tendency_properties = {
        'air_temperature': {
            'dims_like': 'air_temperature',
            'units': 'degK/s',
        }
    }

In tendency_properties, the quantity names specify the quantities for which
tendencies are given. The ``units`` are the units of the output value, which
is also put in the output :py:class:`~sympl.DataArray` as the ``units``
attribute.

``dims_like`` is telling you that the output array will have the same dimensions
as the array you gave it for ``air_temperature`` as an input. If you pass it
an ``air_temperature`` array with ('latitude', 'longitude', 'mid_levels') as
its axes, it will return an array with ('latitude', 'longitude', 'mid_levels')
for the temperature tendency. If ``dims_like`` is not specified in the
``tendency_properties`` dictionary, it is assumed to be the matching quantity
in the input, but for the other quantities ``dims_like`` must always be
explicitly defined. For instance, if the object as a ``diagnostic_properties``
equal to:

.. code-block:: python

    diagnostic_properties = {
        'cloud_fraction': {
            'dims_like': 'air_temperature',
            'units': '',
        }
    }

that the object will output ``cloud_fraction`` in its diagnostics on the
same grid as ``air_temperature``, in dimensionless units.

ImplicitPrognostic
------------------

.. warning:: This component type should be avoided unless you know you need it,
             for reasons discussed in this section.

In addition to the component types described above, computation may be performed by a
:py:class:`~sympl.ImplicitPrognostic`. This class should be avoided unless you
know what you are doing, but it may be necessary in certain cases. An
:py:class:`~sympl.ImplicitPrognostic`, like a :py:class:`~sympl.Prognostic`,
calculates tendencies, but it does so using both the model state and a timestep.
Certain components, like ones handling advection using a spectral method, may
need to derive tendencies from an :py:class:`~sympl.Implicit` object by
representing it using an :py:class:`~sympl.ImplicitPrognostic`.

The reason to avoid using an :py:class:`~sympl.ImplicitPrognostic` is that if
a component requires a timestep, it is making internal assumptions about how
you are timestepping. For example, it may use the timestep to ensure that all
supersaturated water is condensed by the end of the timestep using an assumption
about the timestepping. However, if you use a :py:class:`~sympl.TimeStepper`
which does not obey those assumptions, you may get unintended behavior, such as
some supersaturated water remaining, or too much water being condensed.

For this reason, the :py:class:`~sympl.TimeStepper` objects included in Sympl
do not wrap :py:class:`~sympl.ImplicitPrognostic` components. If you would like
to use this type of component, and know what you are doing, it is pretty easy
to write your own :py:class:`~sympl.TimeStepper` to do so (you can base the code
off of the code in Sympl), or the model you are using might already have
components to do this for you.

If you are wrapping a parameterization and notice that it needs a timestep to
compute its tendencies, that is likely *not* a good reason to write an
:py:class:`~sympl.ImplicitPrognostic`. If at all possible you should modify the
code to compute the value at the next timestep, and write an
:py:class:`~sympl.Implicit` component. You are welcome to reach out to the
developers of Sympl if you would like advice on your specific situation! We're
always excited about new wrapped components.

.. autoclass:: sympl.ImplicitPrognostic
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

.. _Python documentation for dicts: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
