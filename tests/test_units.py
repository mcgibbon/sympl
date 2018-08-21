from sympl import units_are_same, units_are_compatible, is_valid_unit, DataArray, set_units_backend
import numpy as np
import unittest

identical_unit_lists = (
    ('m', 'meter', 'meters'),
    ('s', 'second', 'seconds'),
    ('km', 'kilometers'),
    ('kg/kg', 'g/g'),
    ('degree_Celsius', 'degrees_Celsius', 'degC'),
    ('percent', '%'),
    ('C', 'coulomb'),
    ('degree_east', 'degrees_east', 'degree_East', 'degrees_East',
     'degree_E', 'degrees_E', 'degreeE', 'degreesE'),
    ('degree_north', 'degrees_north', 'degree_North', 'degrees_North',
     'degree_N', 'degrees_N', 'degreeN', 'degreesN'),
)


data_array_conversion_lists = (
    (
        DataArray(
            np.ones([10]),
            attrs={'units': 'm'}
        ),
        DataArray(
            np.ones([10]) * 0.001,
            attrs={'units': 'km'},
        ),
        DataArray(
            np.ones([10]) * 100.,
            attrs={'units': 'centimeters'},
        ),
    ),
    (
        DataArray(
            np.ones([2, 3, 4]) * 273.15,
            attrs={'units': 'degK'},
        ),
        DataArray(
            np.zeros([2, 3, 4]),
            attrs={'units': 'degrees_Celsius'},
        ),
        DataArray(
            np.ones([2, 3, 4]) * 32.,
            attrs={'units': 'degF'},
        )
    )
)


incompatible_unit_lists = (
    ('m', 's', 'degK'),
    ('W/m^2', 'N/cm^2', 'km'),
    ('degree_N', 'degree_E'),
)


valid_units = set()
for L in identical_unit_lists:
    valid_units.update(L)
for L in data_array_conversion_lists:
    valid_units.update([data_array.attrs['units'] for data_array in L])

invalid_unit_names = (
    'george', 'boop', 'degree_Celcius',
)


class UnitFunctionalityTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_is_valid_unit(self):
        for unit_name in valid_units:
            assert is_valid_unit(unit_name), unit_name

    def test_conversion_to_same_units(self):
        for L in identical_unit_lists:
            for unit1 in L:
                for unit2 in L:
                    da1 = DataArray(
                        np.ones([10]),
                        attrs={'units': unit1}
                    )
                    da2 = da1.to_units(unit2)
                    assert np.all(da1.values == da2.values)
                    assert np.byte_bounds(da1.values) == np.byte_bounds(da2.values)

    def test_units_are_compatible(self):
        for L in data_array_conversion_lists:
            for array1 in L:
                for array2 in L:
                    unit1 = array1.attrs['units']
                    unit2 = array2.attrs['units']
                    assert units_are_compatible(unit1, unit2), (unit1, unit2)

    def test_units_are_same(self):
        for L in identical_unit_lists:
            for i, unit1 in enumerate(L):
                for unit2 in L[i:]:
                    assert units_are_same(unit1, unit2), (unit1, unit2)

    def test_is_valid_unit_invalid_values(self):
        for unit_name in invalid_unit_names:
            assert not is_valid_unit(unit_name)


# class PintTests(UnitFunctionalityTests):
#
#     def setUp(self):
#         set_units_backend('pint')


class CFUnitsTests(UnitFunctionalityTests):

    def setUp(self):
        set_units_backend('cfunits')
