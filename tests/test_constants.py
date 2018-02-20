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


if __name__ == '__main__':
    pytest.main([__file__])
