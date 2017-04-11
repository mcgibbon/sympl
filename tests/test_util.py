import pytest
from sympl import (
    UpdateFrequencyWrapper, Prognostic, replace_none_with_default,
    default_constants, ensure_no_shared_keys, SharedKeyError, DataArray,
    combine_dimensions, set_dimension_names,
    TendencyInDiagnosticsWrapper, get_numpy_array,
    restore_dimensions)
from sympl._core.util import update_dict_by_adding_another
from datetime import datetime, timedelta
import numpy as np
import unittest

def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockPrognostic(Prognostic):

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state):
        self._num_updates += 1
        return {}, {'num_updates': self._num_updates}


def test_set_prognostic_update_frequency_calls_initially():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic(state)
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_caches_result():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic(state)
    tendencies, diagnostics = prognostic(state)
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_caches_result_with_datetime():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': datetime(2000, 1, 1)}
    tendencies, diagnostics = prognostic(state)
    tendencies, diagnostics = prognostic(state)
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_updates_result_when_equal():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic({'time': timedelta(hours=0)})
    tendencies, diagnostics = prognostic({'time': timedelta(hours=1)})
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 2


def test_set_prognostic_update_frequency_updates_result_when_greater():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic({'time': timedelta(hours=0)})
    tendencies, diagnostics = prognostic({'time': timedelta(hours=2)})
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 2


def test_replace_none_with_default_replaces_none():
    value = replace_none_with_default('gas_constant_of_dry_air', None)
    assert value is not None


def test_replace_none_with_default_does_not_replace_value():
    value = replace_none_with_default('gas_constant_of_dry_air', -1.)
    assert value == -1.


def test_replace_none_with_default_uses_new_default_constant():
    try:
        default_constants['foo'] = 5.
        value = replace_none_with_default('foo', None)
        assert value == 5.
    finally:  # make sure we restore default_constants after
        default_constants.pop('foo')


def test_replace_none_with_default_uses_replaced_default_constant():
    old_value = default_constants['gas_constant_of_dry_air']
    try:
        default_constants['gas_constant_of_dry_air'] = -5.
        value = replace_none_with_default('gas_constant_of_dry_air', None)
        assert value == -5.
    finally:  # make sure we restore default_constants after
        default_constants['gas_constant_of_dry_air'] = old_value


def test_replace_none_with_default_raises_keyerror_on_missing_value():
    assert 'arglebargle' not in default_constants
    try:
        value = replace_none_with_default('arglebargle', None)
    except KeyError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('No error was raised, but expected KeyError')


def test_update_dict_by_adding_another_adds_shared_arrays():
    old_a = np.array([1., 1.])
    dict1 = {'a': old_a}
    dict2 = {'a': np.array([2., 3.]), 'b': np.array([0., 1.])}
    update_dict_by_adding_another(dict1, dict2)
    assert 'b' in dict1.keys()
    assert dict1['a'] is old_a
    assert np.all(dict1['a'] == np.array([3., 4.]))
    assert np.all(dict2['a'] == np.array([2., 3.]))
    assert np.all(dict2['b'] == np.array([0., 1.]))
    assert len(dict1.keys()) == 2
    assert len(dict2.keys()) == 2


def test_update_dict_by_adding_another_adds_shared_arrays_reversed():
    old_a = np.array([1., 1.])
    dict1 = {'a': np.array([2., 3.])}
    dict2 = {'a': old_a, 'b': np.array([0., 1.])}
    update_dict_by_adding_another(dict2, dict1)
    assert 'b' not in dict1.keys()
    assert dict2['a'] is old_a
    assert np.all(dict2['a'] == np.array([3., 4.]))
    assert np.all(dict1['a'] == np.array([2., 3.]))
    assert np.all(dict2['b'] == np.array([0., 1.]))
    assert len(dict1.keys()) == 1
    assert len(dict2.keys()) == 2


def test_ensure_no_shared_keys_empty_dicts():
    ensure_no_shared_keys({}, {})


def test_ensure_no_shared_keys_one_empty_dict():
    ensure_no_shared_keys({'a': 1, 'b': 2}, {})
    ensure_no_shared_keys({}, {'a': 1, 'b': 2})


def test_ensure_no_shared_keys_with_no_shared_keys():
    ensure_no_shared_keys({'a': 1, 'b': 2}, {'c': 2, 'd': 1})
    ensure_no_shared_keys({'c': 2, 'd': 1}, {'a': 1, 'b': 2})


def test_ensure_no_shared_keys_with_shared_keys():
    try:
        ensure_no_shared_keys({'a': 1, 'b': 2}, {'e': 2, 'a': 1})
    except SharedKeyError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError(
            'No exception raised but expected SharedKeyError.')


class CombineDimensionsTests(unittest.TestCase):

    def setUp(self):
        self.array_1d = DataArray(np.zeros((2,)), dims=['lon'])
        self.array_2d = DataArray(np.zeros((2, 2)), dims=['lat', 'lon'])
        self.array_3d = DataArray(np.zeros((2, 2, 2)),
                                  dims=['lon', 'lat', 'interface_levels'])
        set_dimension_names(
            x=['lon'], y=['lat'], z=['mid_levels', 'interface_levels'])

    def tearDown(self):
        set_dimension_names(x=[], y=[], z=[])

    def test_combine_dimensions_2d_and_3d(self):
        dims = combine_dimensions(
            [self.array_2d, self.array_3d], out_dims=('x', 'y', 'z'))
        assert same_list(dims, ['lon', 'lat', 'interface_levels'])

    def test_combine_dimensions_2d_and_3d_z_y_x(self):
        dims = combine_dimensions(
            [self.array_2d, self.array_3d], out_dims=('z', 'y', 'x'))
        assert same_list(dims, ['interface_levels', 'lat', 'lon'])

    def combine_dimensions_1d_shared(self):
        dims = combine_dimensions(
            [self.array_1d, self.array_1d], out_dims=['x'])
        assert same_list(dims, ['lon'])

    def combine_dimensions_1d_not_shared(self):
        array_1d_x = DataArray(np.zeros((2,)), dims=['lon'])
        array_1d_y = DataArray(np.zeros((2,)), dims=['lat'])
        dims = combine_dimensions([array_1d_x, array_1d_y], out_dims=['x', 'y'])
        assert same_list(dims, ['lon', 'lat'])

    def combine_dimensions_1d_wrong_direction(self):
        try:
            combine_dimensions(
                [self.array_1d, self.array_1d], out_dims=['z'])
        except ValueError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError('No exception raised but expected ValueError.')

    def combine_dimensions_1d_and_2d_extra_direction(self):
        try:
            combine_dimensions(
                [self.array_1d, self.array_2d], out_dims=['y'])
        except ValueError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError('No exception raised but expected ValueError.')


def test_put_prognostic_tendency_in_diagnostics_no_tendencies():
    class MockPrognostic(Prognostic):
        def __call__(self, state):
            return {}, {}

    prognostic = TendencyInDiagnosticsWrapper(MockPrognostic(), 'scheme')
    tendencies, diagnostics = prognostic({})
    assert len(tendencies) == 0
    assert len(diagnostics) == 0


def test_put_prognostic_tendency_in_diagnostics_one_tendency():
    class MockPrognostic(Prognostic):
        tendency_properties = {'quantity': {}}
        def __call__(self, state):
            return {'quantity': 1.}, {}

    prognostic = TendencyInDiagnosticsWrapper(MockPrognostic(), 'scheme')
    tendencies, diagnostics = prognostic({})
    assert 'tendency_of_quantity_due_to_scheme' in prognostic.diagnostics
    tendencies, diagnostics = prognostic({})
    assert 'tendency_of_quantity_due_to_scheme' in diagnostics.keys()
    assert len(diagnostics) == 1
    assert tendencies['quantity'] == 1.
    assert diagnostics['tendency_of_quantity_due_to_scheme'] == 1.

if __name__ == '__main__':
    pytest.main([__file__])
