from sympl import units_are_same, units_are_compatible, is_valid_unit


def test_is_valid_unit_meters():
    assert is_valid_unit('m')
    assert is_valid_unit('meter')
    assert is_valid_unit('meters')


def test_units_are_compatible_meters():
    assert units_are_compatible('m', 'km')
    assert units_are_compatible('kilometers', 'cm')
    assert not units_are_compatible('m', 'm/s')


def test_units_are_same_meters():
    assert units_are_same('m', 'meter')
    assert units_are_same('meters', 'm')
    assert units_are_same('kilometers', 'km')


def test_is_valid_unit_invalid_values():
    assert not is_valid_unit('george')
    assert not is_valid_unit('boop')
