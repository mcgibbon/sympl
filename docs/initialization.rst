========================
State and Initialization
========================

In a Sympl model, physical quantities are stored in a state dictionary. This is
a Python **dict** with keys that are strings, indicating the quantity name, and
values are :py:class:`sympl.DataArray` objects. The :py:class:`sympl.DataArray`
is a slight modification of the ``DataArray`` object from xarray_. It
maintains attributes when it is on the left hand side of addition or
subtraction, and contains a helpful method for converting units. Any
information about the grid the data is using that components need should be
put as attributes in the ``attrs`` of the ``DataArray`` objects. Deciding on
these attributes (if any) is up to the component developers.

.. _xarray: http://xarray.pydata.org/en/stable/

.. autoclass:: sympl.DataArray
    :members:
    :special-members:
    :exclude-members: __weakref__,__metaclass__

There is one quantity which is not stored as a :py:class:`sympl.DataArray`, and
that is "time". Time is stored as a datetime or timedelta-like object.

Code to initialize the state is not present in Sympl, by design, since this
depends heavily on the details of the model you are running. You may find
helper functions to do this in model packages, or you can write your own code
to initialize the state. For example, below you can see code to initialize
a state with random temperature and pressure on a lat-lon grid (random values
are used for demonstration purposes only, and are not recommended in a real
model).

.. code-block::python

    from datetime import datetime
    import numpy as np
    from sympl import DataArray, set_dimension_names
    n_lat = 64
    n_lon = 128
    n_height = 32
    set_dimension_names(x='lat', y='lon', z=('mid_levels', 'interface_levels'))
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

The call to :py:function:`sympl.set_dimension_names` tells the framework
what dimension names correspond to what directions. This information is used
by components to make sure the axes are in the right order.

Choice of Datetime
------------------

The built-in ``datetime`` object in Python (as used above) assumes the
proleptic Gregorian calendar, which extends the Gregorian calendar back
infinitely. If you want to use a different type of calendar, you should use
a different datetime object, but one which has the same interface as the Python
object.

We think and hope the netcdftime_ package will have objects with this
functionality. Once it's available on pypi and documented, update this doc
page to describe how to use it!

.. _netcdftime: https://github.com/Unidata/netcdftime

