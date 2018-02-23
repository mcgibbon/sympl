===========
Model State
===========

In a Sympl model, physical quantities are stored in a state dictionary. This is
a Python **dict** with keys that are strings, indicating the quantity name, and
values are :py:class:`~sympl.DataArray` objects. The :py:class:`~sympl.DataArray`
is a slight modification of the ``DataArray`` object from xarray_. It
maintains attributes when it is on the left hand side of addition or
subtraction, and contains a helpful method for converting units. Any
information about the grid the data is using that components need should be
put as attributes in the ``attrs`` of the ``DataArray`` objects. Deciding on
these attributes (if any) is mostly up to the component developers. However,
in order to use the TimeStepper objects and several helper functions from Sympl,
it is required that a "units" attribute is present.

.. _xarray: http://xarray.pydata.org/en/stable/

.. autoclass:: sympl.DataArray
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

There is one quantity which is not stored as a :py:class:`~sympl.DataArray`, and
that is "time". Time must be stored as a datetime or timedelta-like object.

Code to initialize the state is intentionally not present in Sympl, since this
depends heavily on the details of the model you are running. You may find
helper functions to create an initial state in model packages, or you can write
your own. For example, below you can see code to initialize
a state with random temperature and pressure on a lat-lon grid (random values
are used for demonstration purposes only, and are not recommended in a real
model).

.. code-block:: python

    from datetime import datetime
    import numpy as np
    from sympl import DataArray, add_direction_names
    n_lat = 64
    n_lon = 128
    n_height = 32
    add_direction_names(x='lat', y='lon', z=('mid_levels', 'interface_levels'))
    state = {
        "time": datetime(2000, 1, 1),
        "air_temperature": DataArray(
            np.random.rand(n_lat, n_lon, n_height),
            dims=('lat', 'lon', 'mid_levels'),
            attrs={'units': 'degK'}),
        "air_pressure": DataArray(
            np.random.rand(n_lat, n_lon, n_height),
            dims=('lat', 'lon', 'mid_levels'),
            attrs={'units': 'Pa'}),
        "air_pressure_on_interface_levels": DataArray(
            np.random.rand(n_lat, n_lon, n_height + 1),
            dims=('lat', 'lon', 'interface_levels'),
            attrs=('units': 'Pa')),
        }

The call to :py:func:`~sympl.add_direction_names` tells Sympl
what dimension names correspond to what directions. This information is used
by components to make sure the axes are in the right order.

Choice of Datetime
------------------

The built-in ``datetime`` object in Python (as used above) assumes the
proleptic Gregorian calendar, which extends the Gregorian calendar back
infinitely. Sympl provides a :py:func:`~sympl.datetime` function which returns
a datetime-like object, and allows a variety of different calendars. If a
calendar other than 'proleptic_gregorian' is specified, one of the classes from
the netcdftime_ package will be used. Of course, this requires that it is
installed! If it's not, you will get an error, and should ``pip install netcdftime``.
Sympl also includes :py:class:`~sympl.timedelta` for convenience. This is just
the default Python ``timedelta``.

To repeat, the calendar your model is using depends entirely on what
object you're using to store time in the state dictionary, and the default one
uses the proleptic Gregorian calendar used by the default Python ``datetime``.

.. autofunction:: sympl.datetime

.. autoclass:: sympl.timedelta

.. _netcdftime: https://github.com/Unidata/netcdftime

Naming Quantities
-----------------

If you are a model user, the names of your quantities should coincide with the
names used by the components you are using in your model. Basically, the
components you are using dictate what quantity names you must use. If you are
a model developer, we have a set of guidelines for naming quantities.

.. note:: The following is intended for model developers.

All quantity names should be verbose, and fully descriptive. Within a
component you can set a quantity to an abbreviated variable, such as

.. code-block:: python

    theta = state['air_potential_temperature']

This ensures that your code is self-documenting. It is immediately apparent
to anyone reading your code that theta refers to potential temperature of air,
even if they are not familiar with theta as a common abbreviation.

We strongly recommend using the standard names according to `CF conventions`_.
In addition to making sure your code is self-documenting, this helps make sure
that different components are compatible with one another, since they all
need to use the same name for a given quantity in the model state.

If your quantity is on vertical interface levels, you should name it using
the form "<name>_on_interface_levels". If this is not specified, it is
assumed that the quantity is on vertical mid levels. This is necessary
because the same quantity may be specified on both mid and interface levels
in the same model state.

When in doubt about names, look at what other components have been written that
use the same quantity. If it looks like their name is verbose and follows the
`CF conventions`_ then you should probably use the same name.

.. _`CF conventions`: http://cfconventions.org/standard-names.html
