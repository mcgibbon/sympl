from sympl import set_constant, default_constants
import pytest


def test_setting_existing_constant():

    set_constant('seconds_per_day', 100000, 'seconds/day')

    new_constant = default_constants['seconds_per_day']
    assert new_constant.values == 100000
    assert new_constant.units == 'seconds/day'


def test_setting_new_constant():

    set_constant('my_own_constant', 10., 'W m^-1 degK^-1')

    assert 'my_own_constant' in default_constants

    new_constant = default_constants['my_own_constant']
    assert new_constant.values == 10.
    assert new_constant.units == 'W m^-1 degK^-1'


def test_setting_wrong_units():

    with pytest.raises(ValueError) as excinfo:
        set_constant('abcd', 100, 'Wii')

    assert 'valid unit' in str(excinfo.value)
