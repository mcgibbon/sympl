from ._pint import (
    backend_units_are_same, backend_is_valid_unit, backend_clean_units,
    backend_array_from_units_to_another
)
from . import _cfunits
from ..exceptions import UnitError
import numpy as np


def units_are_same(unit1, unit2):
    return backend_units_are_same(unit1, unit2)


def is_valid_unit(unit_string):
    return backend_is_valid_unit(unit_string)


def clean_units(unit_string):
    return backend_clean_units(unit_string)


def array_from_units_to_another(value, original_units, new_units):
    return backend_array_from_units_to_another(value, original_units, new_units)


def units_are_compatible(unit1, unit2):
    """
    Determine whether a unit can be converted to another unit.

    Parameters
    ----------
    unit1 : str
    unit2 : str

    Returns
    -------
    units_are_compatible : bool
        True if the first unit can be converted to the second unit.
    """
    try:
        array_from_units_to_another(np.array(1.), unit1, unit2)
    except UnitError:
        return False
    return True


def data_array_to_array_with_units(data_array, out_units):
    if not hasattr(data_array, 'attrs') or 'units' not in data_array.attrs:
        raise TypeError(
            'Cannot retrieve units from type {}'.format(type(data_array)))
    elif units_are_same(data_array.attrs['units'], out_units):
        out_array = data_array.values
    elif not units_are_compatible(data_array.attrs['units'], out_units):
        raise UnitError(
            'Units {} and {} are incompatible'.format(
                data_array.attrs['units'], out_units)
        )
    else:
        out_array = array_from_units_to_another(
            data_array.values, data_array.attrs['units'], out_units)
    return out_array


def set_backend(backend_name):
    global backend_units_are_same, backend_is_valid_unit, backend_clean_units
    global backend_array_from_units_to_another
    if backend_name.lower() == 'pint':
        from ._pint import (
            backend_units_are_same, backend_is_valid_unit, backend_clean_units,
            backend_array_from_units_to_another
        )
    elif backend_name.lower() in ('cfunits', 'cf-units', 'cf_units'):
        from ._cfunits import (
            backend_units_are_same, backend_is_valid_unit, backend_clean_units,
            backend_array_from_units_to_another
        )

if _cfunits.cf_units is not None:
    set_backend('cfunits')
