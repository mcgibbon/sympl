from sympl import set_constant
import pytest


def test_setting_wrong_units():

    with pytest.raises(TypeError) as excinfo:
        set_constant('abcd', 100, 'Wii')

    assert 'valid unit' in str(excinfo.value)
