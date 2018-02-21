from sympl import get_constant, set_constant, DataArray
from sympl._core.constants import constants
from sympl._core.units import is_valid_unit
import pytest


def test_constants_are_dataarray():
    for constant_name, value in constants.items():
        assert isinstance(value, DataArray), constant_name


def test_constants_have_valid_units():
    for constant_name, value in constants.items():
        assert 'units' in value.attrs, constant_name
        assert is_valid_unit(value.attrs['units']), constant_name


def test_setting_existing_constant():

    set_constant('seconds_per_day', 100000, 'seconds/day')

    new_constant = constants['seconds_per_day']
    assert new_constant.values == 100000
    assert new_constant.units == 'seconds/day'


def test_setting_new_constant():

    set_constant('my_own_constant', 10., 'W m^-1 degK^-1')

    assert 'my_own_constant' in default_constants

    new_constant = constants['my_own_constant']
    assert new_constant.values == 10.
    assert new_constant.units == 'W m^-1 degK^-1'


def test_setting_wrong_units():

    with pytest.raises(ValueError) as excinfo:
        set_constant('abcd', 100, 'Wii')

    assert 'valid unit' in str(excinfo.value)

if __name__ == '__main__':
    pytest.main([__file__])
