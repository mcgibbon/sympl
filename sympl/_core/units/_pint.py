# -*- coding: utf-8 -*-
import pint


class UnitRegistry(pint.UnitRegistry):

    def __call__(self, input_string, **kwargs):
        return super(UnitRegistry, self).__call__(
            input_string.replace(
                u'%', 'percent').replace(
                u'°', 'degree'
            ),
            **kwargs)


unit_registry = UnitRegistry()
unit_registry.define('degrees_north = degree_north = degree_N = degrees_N = degreeN = degreesN')
unit_registry.define('degrees_east = degree_east = degree_E = degrees_E = degreeE = degreesE')
unit_registry.define('degrees_Celsius = degC')
unit_registry.define('degree_Celsius = degC')
unit_registry.define('percent = 0.01*count = %')


def backend_units_are_compatible(unit1, unit2):
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
        unit_registry(unit1).to(unit2)
        return True
    except pint.errors.DimensionalityError:
        return False


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
    return unit_registry(unit1) == unit_registry(unit2)


def backend_clean_units(unit_string):
    return str(unit_registry(unit_string).to_base_units().units)


def backend_is_valid_unit(unit_string):
    """Returns True if the unit string is recognized, and False otherwise."""
    unit_string = unit_string.replace(
        '%', 'percent').replace(
        '°', 'degree')
    try:
        unit_registry(unit_string)
    except pint.UndefinedUnitError:
        return False
    else:
        return True


def backend_array_from_units_to_another(value, original_units, new_units):
    return unit_registry.Quantity(value, original_units).to(new_units).magnitude

