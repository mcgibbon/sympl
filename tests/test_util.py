import pytest
from sympl import (
    set_prognostic_update_frequency, Prognostic, replace_none_with_default,
    default_constants, ensure_no_shared_keys, SharedKeyException)
from sympl._core.util import update_dict_by_adding_another
from datetime import datetime, timedelta
import numpy as np


class MockPrognostic(Prognostic):

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state):
        self._num_updates += 1
        return {}, {'num_updates': self._num_updates}


def test_set_prognostic_update_frequency_calls_initially():
    set_prognostic_update_frequency(MockPrognostic, timedelta(hours=1))
    prognostic = MockPrognostic()
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic(state)
    assert len(tendencies) == 0
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_caches_result():
    set_prognostic_update_frequency(MockPrognostic, timedelta(hours=1))
    prognostic = MockPrognostic()
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic(state)
    tendencies, diagnostics = prognostic(state)
    assert len(tendencies) == 0
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_caches_result_with_datetime():
    set_prognostic_update_frequency(MockPrognostic, timedelta(hours=1))
    prognostic = MockPrognostic()
    state = {'time': datetime(2000, 1, 1)}
    tendencies, diagnostics = prognostic(state)
    tendencies, diagnostics = prognostic(state)
    assert len(tendencies) == 0
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_updates_result_when_equal():
    set_prognostic_update_frequency(MockPrognostic, timedelta(hours=1))
    prognostic = MockPrognostic()
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic({'time': timedelta(hours=0)})
    tendencies, diagnostics = prognostic({'time': timedelta(hours=1)})
    assert len(tendencies) == 0
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 2


def test_set_prognostic_update_frequency_updates_result_when_greater():
    set_prognostic_update_frequency(MockPrognostic, timedelta(hours=1))
    prognostic = MockPrognostic()
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic({'time': timedelta(hours=0)})
    tendencies, diagnostics = prognostic({'time': timedelta(hours=2)})
    assert len(tendencies) == 0
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
    except SharedKeyException:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError(
            'No exception raised but expected SharedKeyException.')


if __name__ == '__main__':
    pytest.main([__file__])
