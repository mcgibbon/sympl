from cf_units import Unit
try:
    import cf_units
except ImportError:
    cf_units = None
from ..exceptions import UnitError


def backend_units_are_same(unit1, unit2):
    """
    Compare two unit strings for equality.

    Parameters
    ----------
    unit1 : str
    unit2 : str

    Returns
    -------
    units_are_same : bool
        True if the two input unit strings represent the same unit.
    """
    return Unit(unit1) == Unit(unit2)


def backend_clean_units(unit_string):
    return Unit(unit_string).format(cf_units.UT_NAMES)


def backend_is_valid_unit(unit_string):
    """Returns True if the unit string is recognized, and False otherwise."""
    try:
        Unit(unit_string)
    except ValueError:
        return False
    return True


def backend_array_from_units_to_another(value, original_units, new_units):
    try:
        return Unit(original_units).convert(value, Unit(new_units))
    except ValueError:
        raise UnitError('units {} and {} are incompatible'.format(
            original_units, new_units)
        )
