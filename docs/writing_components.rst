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

Let's start with a Prognostic component which relaxes temperature towards some
target temperature.

.. code-block:: python

    from sympl import (
        Prognostic, get_numpy_arrays_with_properties,
        restore_data_arrays_with_properties)

    class TemperatureRelaxation(Prognostic):

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

        def __init__(self, tau=1., target_temperature=300.):
            self._tau = tau
            self._T0 = target_temperature

        def __call__(self, state):
            # we get numpy arrays with specifications from input_properties
            raw_arrays = get_numpy_arrays_with_properties(
                state, self.input_properties)
            T = raw_arrays['air_temperature']
            # here the actual computation happens
            raw_tendencies = {
                'air_temperature': (T - self._T0)/self._tau,
            }
            # now we re-format the data in a way the host model can use
            diagnostics = {}
            tendencies = restore_data_arrays_with_properties(
                raw_tendencies, self.tendency_properties,
                state, self.input_properties)
            return tendencies, diagnostics

Imports
*******

There are a lot of parts to that code, so let's go through some of them
step-by-step. First we have to import objects and functions from Sympl that
we plan to use. The import statement should always go at the top of your file
so that it can be found right away by anyone reading your code.

.. code-block:: python

    from sympl import (
        Prognostic, get_numpy_arrays_with_properties,
        restore_data_arrays_with_properties)

Define an Object
****************

Once these are imported, there's this line:

.. code-block:: python

    class TemperatureRelaxation(Prognostic):

This is the syntax for defining an object in Python. ``TemperatureRelaxation``
will be the name of the new object. The :py:class:`~sympl.Prognostic`
in parentheses is telling Python that ``TemperatureRelaxation`` is a *subclass* of
:py:class:`~sympl.Prognostic`. This tells Sympl that it can expect your object
to behave like a :py:class:`~sympl.Prognostic`.

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
can also understand these properties, and use them to automatically
acquire arrays in the dimensions and units that you need.
It can also test thatsome of these properties are accurate.
It's your responsibility, though, to make sure that the input units are the
units you want to acquire in the numpy array data, and that the output units
are the units of the values in the raw output arrays that you want to convert
to :py:class:`~sympl.DataArray` objects.

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

        def __init__(self, damping_timescale=1., target_temperature=300.):
            """
            damping_timescale is the damping timescale in seconds.
            target_temperature is the temperature that will be relaxed to,
            in degrees Kelvin.
            """
            self._tau = damping_timescale
            self._T0 = target_temperature

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

            self._tau = damping_timescale
            self._T0 = target_temperature

Obviously ``self._tau`` is the damping timescale, and ``self._T0`` is the
target temperature for the relaxation. Now you can use these shorter variables
in the actual code to keep long lines for equations short, knowing that your
variables are well-documented.

The Computation
***************

That brings us to the ``__call__`` method. This is what's called when you
use the object as though it is a function. In Sympl components, this is the
method which takes in a state dictionary and returns dictionaries with outputs.

.. code-block:: python

        def __call__(self, state):
            # we get numpy arrays with specifications from input_properties
            raw_arrays = get_numpy_arrays_with_properties(
                state, self.input_properties)
            T = raw_arrays['air_temperature']
            # here the actual computation happens
            raw_tendencies = {
                'air_temperature': (T - self._T0)/self._tau,
            }
            # now we re-format the data in a way the host model can use
            diagnostics = {}
            tendencies = restore_data_arrays_with_properties(
                raw_tendencies, self.tendency_properties,
                state, self.input_properties)
            return diagnostics, tendencies

There are two helper functions used in this code that we strongly recommend
using. They take care of the work of making sure you get variables that are
in the units your component needs, and have the dimensions your component needs.

:py:func:`~sympl.get_numpy_arrays_with_properties` uses the input_properties
dictionary you give it to extract numpy arrays with those properties from the
input state. It will convert units to ensure the numbers are in the specified
units, and it will reshape the data to give it the shape specified in ``dims``.
For example, if dims is ``['*', 'z']`` then it will give you a 2-dimensional array
whose second axis is the vertical, and first axis is a flattening of any other
dimensions. If you specify ``['*', 'mid_levels']`` then the result is similar, but
only 'mid_levels' is an acceptable vertical dimension. The ``match_dims_like``
property on ``air_pressure`` tells Sympl that any wildcard-matched dimensions
(ones that match 'x', 'y', 'z', or '*') should be the same between the two
quantities, meaning they're on the same grid for those wildcards. You can still,
however, have one be on say 'mid_levels' and another on 'interface_levels' if
those dimensions are explicitly listed (instead of listing 'z').

:py:func:`~sympl.restore_data_arrays_with_properties` does something fairly
magical. In this example, it takes the raw_tendencies dictionary and converts
the value for 'air_temperature' from a numpy array to a DataArray that has
the same dimensions as ``air_temperature`` had in the input state. That means
that you could pass this object a state with whatever dimensions you want,
whether it's (x, y, z), or (z, x, y), or (x, y), or (station_number, z), etc.
and this component will be able to take in that state, and return a
tendency dictionary with the same dimensions (and order) that the model uses!
And internally you can work with a simple 1-dimensional array. This is
particularly useful for writing pointwise components using ``['*']`` or column
components with ``['*', 'z']`` or ``['z', '*']``.

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

        def __call__(self, state):
            # we get numpy arrays with specifications from input_properties
            raw_arrays = get_numpy_arrays_with_properties(
                state, self.input_properties)
            T = raw_arrays['T']
            # here the actual computation happens
            raw_tendencies = {
                'T': (T - self._T0)/self._tau,
            }
            # now we re-format the data in a way the host model can use
            diagnostics = {}
            tendencies = restore_data_arrays_with_properties(
                raw_tendencies, self.tendency_properties,
                state, self.input_properties)
            return diagnostics, tendencies

Instead of using 'air_temperature' in the raw_arrays and raw_tendencies
dictionaries, we can use 'T'. This doesn't matter much for a name as short as
air_temperature, but it might matter for longer names like
'correlation_of_eastward_wind_and_liquid_water_potential_temperature_on_interface_levels'.

Also notice that even though the alias is set in input_properties, it is also
used when restoring DataArrays. If there is an output that is not
also an input, the alias could instead be set in ``diagnostic_properties``,
``tendency_properties``, or ``output_properties``, wherever is relevant.
