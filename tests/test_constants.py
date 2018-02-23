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
    new_constant = get_constant('seconds_per_day', units='seconds/day')
    assert new_constant == 100000


def test_setting_new_constant():

    set_constant('my_own_constant', 10., 'W m^-1 degK^-1')
    new_constant = get_constant('my_own_constant', units='W m^-1 degK^-1')
    assert new_constant == 10.


def test_converting_existing_constant():
    g_m_per_second = get_constant('gravitational_acceleration', 'm s^-2')
    g_km_per_second = get_constant('gravitational_acceleration', 'km s^-2')
    assert g_km_per_second == g_m_per_second * 0.001


def test_setting_wrong_units():

    with pytest.raises(ValueError) as excinfo:
        set_constant('abcd', 100, 'Wii')

    assert 'valid unit' in str(excinfo.value)

if __name__ == '__main__':
    pytest.main([__file__])
