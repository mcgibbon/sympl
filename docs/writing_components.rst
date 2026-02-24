==================
Writing Components
==================

.. note:: This section is intended for model developers. If you intend
    to use only components that are already written, you can probably ignore it.

Perhaps the best way to learn how to write components is to read components
someone else has written. For example, you can look at the CliMT project. Here
we will go over a couple examples of physically simple, made-up components
to talk about the parts of their code.

Writing an Example
------------------

Let's start with a TendencyComponent component which relaxes temperature towards some
target temperature. We'll go over the sections of this example step-by-step
below.

.. code-block:: python

    from sympl import (
        TendencyComponent, get_numpy_arrays_with_properties,
        restore_data_arrays_with_properties)

    class TemperatureRelaxation(TendencyComponent):

        input_properties = {
            'air_temperature': {
                'dims': ['*'],
                'units': 'degK',
            },
            'vertical_wind': {
                'dims': ['*'],
                'units': 'm/s',
                'match_dims_like': ['air_temperature']
            }
        }

        diagnostic_properties = {}

        tendency_properties = {
            'air_temperature': {
                'dims_like': 'air_temperature',
                'units': 'degK/s',
            }
        }

        def __init__(self, damping_timescale_seconds=1., target_temperature_K=300.):
            self._tau = damping_timescale_seconds
            self._T0 = target_temperature_K

        def array_call(self, state):
            tendencies = {
                'air_temperature': (state['air_temperature'] - self._T0)/self._tau,
            }
            diagnostics = {}
            return tendencies, diagnostics

Imports
*******

There are a lot of parts to that code, so let's go through some of them
step-by-step. First we have to import objects and functions from Sympl that
we plan to use. The import statement should always go at the top of your file
so that it can be found right away by anyone reading your code.

.. code-block:: python

    from sympl import (
        TendencyComponent, get_numpy_arrays_with_properties,
        restore_data_arrays_with_properties)

Define an Object
****************

Once these are imported, there's this line:

.. code-block:: python

    class TemperatureRelaxation(TendencyComponent):

This is the syntax for defining an object in Python. ``TemperatureRelaxation``
will be the name of the new object. The :py:class:`~sympl.TendencyComponent`
in parentheses is telling Python that ``TemperatureRelaxation`` is a *subclass* of
:py:class:`~sympl.TendencyComponent`. This tells Sympl that it can expect your object
to behave like a :py:class:`~sympl.TendencyComponent`.

Define Attributes
*****************

The next few lines define attributes of your object:

.. code-block:: python

        input_properties = {
            'air_temperature': {
                'dims': ['*'],
                'units': 'degK',
            },
            'eastward_wind': {
                'dims': ['*'],
                'units': 'm/s',
                'match_dims_like': ['air_temperature']
            }
        }

        diagnostic_properties = {}

        tendency_properties = {
            'air_temperature': {
                'dims_like': 'air_temperature',
                'units': 'degK/s',
            }
        }

.. note:: 'eastward_wind' wouldn't normally make sense as an input for this object,
          it's only included so we can talk about *match_dims_like*.

These attributes will be attributes both of the class object you're defining
and of any instances of that object. That means you can access them using:

.. code-block:: python

    TemperatureRelaxation.input_properties

or on an instance, as when you do:

.. code-block:: python

    prognostic = TemperatureRelaxation()
    prognostic.input_properties

These properties are described in :ref:`Component Types`. They are very useful!
They clearly document your code. Here we can see that air_temperature will
be used as a 1-dimensional flattened array in units of degrees Kelvin. Sympl
uses these properties to automatically acquire arrays in the dimensions and
units that you need, and to automatically convert your output back into a
form consistent with the dimensions of the model state. It will warn you if
you create extra outputs which are not defined in the properties, or if there
is an output defined in the properties that is missing.

It is possible that some of these attributes won't be known until you
create the object (they may depend on things passed in on initialization).
If that's the case, you can write the ``__init__`` method (see below) so that
it sets any relevant properties like ``self.input_properties`` to have the
correct values.

Initialization Method
*********************

Next we see a method being defined for this class, which may seem to have a
weird name:

.. code-block:: python

        def __init__(self, damping_timescale_seconds=1., target_temperature_K=300.):
            self._tau = damping_timescale_seconds
            self._T0 = target_temperature_K

This is the function that is called when you create an instance of your object.
All methods on objects take in a first argument called ``self``. You don't see
it when you call those methods, it gets added in automatically. ``self`` is
a variable that refers to the object on which the method is being called -
it's the object itself! When you store attributes on self, as we see in this
code, they stay there. You can access them when the object is called later.

Notice some things about the way variables have been named in this ``__init__``.
The parameters are fairly verbose names which almost fully describe what they
are (apart from the units, which are in the documentation string). This is
best because it is entirely clear what these values are when others are using
your object. You write code for people, not computers! Compilers write code for
computers.

Then we take these inputs and store them as attributes with shorter names. This
is also optimal. What these attributes mean is clearly defined in the two lines:

.. code-block:: python

            self._tau = damping_timescale_seconds
            self._T0 = target_temperature_K

Obviously ``self._tau`` is the damping timescale, and ``self._T0`` is the
target temperature for the relaxation. Now you can use these shorter variables
in the actual code to keep long lines for equations short, knowing that your
variables are well-documented.

The Computation
***************

That brings us to the ``array_call`` method. In Sympl components, this is the
method which takes in a state dictionary as numpy arrays (*not* ``DataArray``)
and returns dictionaries with numpy array outputs.

.. code-block:: python

        def array_call(self, state):
            tendencies = {
                'air_temperature': (state['air_temperature'] - self._T0)/self._tau,
            }
            diagnostics = {}
            return tendencies, diagnostics

Sympl will automatically handle taking in the input state of ``DataArray``
objects and converting it to the form defined by the ``input_properties`` of your
component. This is handled by the active :ref:`Array Backend <Array Backends>`.
It will convert units to ensure the numbers are in the specified
units, and it will reshape the data to give it the shape specified in ``dims``.
For example, if dims is ``['*', 'mid_levels']`` then it will give you a
2-dimensional array whose second axis is the vertical on mid levels, and first
axis is a flattening of any other dimensions. The ``match_dims_like``
property on ``air_pressure`` tells Sympl that any wildcard-matched dimensions
('*') should be the same between the two
quantities, meaning they're on the same grid for those wildcards. You can still,
however, have one be on say 'mid_levels' and another on 'interface_levels' if
those dimensions are explicitly listed.

After you return dictionaries of numpy arrays, Sympl will convert these outputs
back to ``DataArray`` objects. In this example, it takes the tendencies
dictionary and converts the value for 'air_temperature' from a numpy array to a
``DataArray`` that has the same dimensions as ``air_temperature`` had in the
input state. That means that you could pass this object a state with whatever
dimensions you want, whether it's ``('longitude', 'latitude', 'mid_levels')``, or
``('interface_levels',)`` or ``('station_number', 'planet_number')``, etc.
and this component will be able to take in that state, and return a
tendency dictionary with the same dimensions (and order) that the model uses!
And internally you can work with a simple 1-dimensional array. This is
particularly useful for writing pointwise components using ``['*']`` or column
components with, for example, ``['*', 'mid_levels']`` or
``['interface_levels', '*']``.

You can read more about properties in the section
:ref:`Input/Output Properties`.

.. autofunction:: sympl.get_numpy_arrays_with_properties

.. autofunction:: sympl.restore_data_arrays_with_properties


Aliases
-------

.. note:: Using aliases isn't necessary, but it may make your code easier to
          read if you have long quantity names

Let's say if instead of the properties we set before, we have


.. code-block:: python

        input_properties = {
            'air_temperature': {
                'dims': ['*'],
                'units': 'degK',
                'alias': 'T',
            },
            'eastward_wind': {
                'dims': ['*'],
                'units': 'm/s',
                'match_dims_like': ['air_temperature']
                'alias': 'u',
            }
        }

The difference here is we've set 'T' and 'u' to be *aliases* for
'air_temperature' and 'eastward_wind'. What does that mean? Well, in the
computational code, we can write:

.. code-block:: python

        def array_call(self, state):
            tendencies = {
                'T': (state['T'] - self._T0)/self._tau,
            }
            diagnostics = {}
            return tendencies, diagnostics

Instead of using 'air_temperature' in the raw_arrays and raw_tendencies
dictionaries, we can use 'T'. This doesn't matter much for a name as short as
air_temperature, but it might matter for longer names like
'correlation_of_eastward_wind_and_liquid_water_potential_temperature_on_interface_levels'.

Also notice that even though the alias is set in input_properties, it is also
used when restoring DataArrays. If there is an output that is not
also an input, the alias could instead be set in ``diagnostic_properties``,
``tendency_properties``, or ``output_properties``, wherever is relevant.


Using Tracers
-------------

.. note:: This feature is mostly used in dynamical cores. If you don't think you need
          this, you probably don't.

Sympl's base components have some features to automatically create tracer arrays
for use by dynamical components. If an :py:class:`~sympl.Stepper`,
:py:class:`~sympl.TendencyComponent`, or :py:class:`~sympl.ImplicitTendencyComponent`
component specifies ``uses_tracers = True`` and sets ``tracer_dims``, this
feature is enabled.

.. code-block:: python

        class MyDynamicalCore(Stepper):

            uses_tracers = True
            tracer_dims = ['tracer', '*', 'mid_levels']

            [...]

``tracer_dims`` is a list or tuple in the form of a ``dims`` attribute on one of its
inputs, and must have a "tracer" dimension. This dimension refers to which
tracer (you could call it "tracer number").

Once this feature is enabled, the ``state`` passed to ``array_call`` on the
component will include a quantity called "tracers" with the dimensions
specified by ``tracer_dims``. It will also be required that these tracers
are used in the output. For a :py:class:`~sympl.Stepper` component, "tracers"
must be present in the output state, and for a :py:class:`~sympl.TendencyComponent` or
:py:class:`~sympl.ImplicitTendencyComponent` component "tracers" must be present in
the tendencies, with the same dimensions as the input "tracers".

On these latter two components, you should also specify a
``tracer_tendency_time_unit`` property, which refers to the time part of the
tendency unit. For example, if the input tracer is in units of ``g m^-3``, and
``tracer_tendency_time_unit`` is "s", then the output tendency will be in
units of ``g m^-3 s^-1``. This value is set as "s" (or seconds) by default.

.. code-block:: python

        class MyDynamicalCore(TendencyComponent):

            uses_tracers = True
            tracer_dims = ['tracer', '*', 'mid_levels']
            tracer_tendency_time_unit = 's'

            [...]
