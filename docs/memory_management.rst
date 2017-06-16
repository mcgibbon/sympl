=================
Memory Management
=================

.. warning:: This section contains fairly advanced topics. If you find it
             confusing, that's because the behavior *is* confusing.

Arrays
------

If possible, you should try to be aware of when there are two code references
to the same in-memory array. This can help avoid some common bugs. Let's start
with an example. Say you create a ConstantPrognostic object like so::

    >>> import numpy as np
    >>> from sympl import ConstantPrognostic, DataArray
    >>> array = DataArray(
            np.ones((5, 5, 10)),
            dims=('lon', 'lat', 'lev'), attrs={'units': 'K/s'})
    >>> tendencies = {'air_temperature': array}
    >>> prognostic = ConstantPrognostic(tendencies)

This is all fine so far. But it's important to know that now ``array`` is the
same array stored inside ``prognostic``::

    >>> out_tendencies, out_diagnostics = prognostic({})
    >>> out_tendencies['air_temperature'] is array  # same place in memory
    True

So if you were to modify ``array``, it would *change the output given by
prognostic*::

    >>> array[:] = array * 5.
    >>> out_tendencies, out_diagnostics = prognostic({})
    >>> out_tendencies['air_temperature'] is array
    True
    >>> np.all(out_tendencies['air_temperature'].values == array.values)
    True

When in doubt, assume that any array you put into a component when it is
initialized should not be modified any more, unless changing the values in the
component is intentional.

However, this code would not modify the array in ``prognostic``::

    >>> array = array * 5.
    >>> out_tendencies, out_diagnostics = prognostic({})
    >>> out_tendencies['air_temperature'] is array
    False
    >>> np.all(out_tendencies['air_temperature'].values == array.values)
    False

What's the difference? We took away the ``[:]`` on the left hand side of the
assignment operator. when ``[:]`` is included, python modifies the array on the
left hand side, but when it's not included it tells the python variable name
"array" to refer to what is on the right hand side. These are subtly different
things - one involves modifying the memory that ``array`` already refers to,
the other involves telling ``array`` to refer to a different place in memory.
More precisely, having ``array =`` tells python
that you want to change what the variable ``array`` refers to, and set it to
be the thing on the right hand side, while ``array[:] =`` tells python to
call the ``__setitem__(key, value)`` method of ``array`` with the contents
of the square parentheses as the key and the right hand side as the value.

Interestingly, ``array = array * 5.`` has different behavior from
``array *= 5.``. The first one will change what ``array`` refers to, as before,
while the second one will modify ``array`` in-place without changing the
reference. Writing ``array *= 5`` is the same as writing ``array[:] = array * 5'``.
All similarly written operations (``-=``, ``+=``, ``/=``, etc.) are
in-place operations.
