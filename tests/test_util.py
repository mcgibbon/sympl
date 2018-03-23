import unittest
import numpy as np
import pytest
from sympl import (
    Prognostic, ensure_no_shared_keys, SharedKeyError, DataArray,
    Implicit, Diagnostic,
    InvalidPropertyDictError)
from sympl._core.util import update_dict_by_adding_another, combine_dims, get_component_aliases


def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockPrognostic(Prognostic):

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state):
        self._num_updates += 1
        return {}, {'num_updates': self._num_updates}


class MockImplicit(Implicit):

    def __init__(self):
        self._a = 1

    def __call__(self, state):
        return self._a


class MockDiagnostic(Diagnostic):

    def __init__(self):
        self._a = 1

    def __call__(self, state):
        return self._a


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


class DummyPrognostic(Prognostic):
    input_properties = {'temperature': {'alias': 'T'}}
    diagnostic_properties = {'pressure': {'alias': 'P'}}
    tendency_properties = {'temperature': {}}

    def __init__(self):
        self._a = 1

    def __call__(self, state):
        return self._a


def test_get_component_aliases_with_no_args():
    aliases = get_component_aliases()
    assert type(aliases) == dict
    assert len(aliases.keys()) == 0


def test_get_component_aliases_with_single_component_arg():
    components = [MockPrognostic(), MockImplicit(), MockDiagnostic()]
    for c, comp in enumerate(components):
        aliases = get_component_aliases(comp)
        assert type(aliases) == dict
        if c == 3:
            assert len(aliases.keys()) == 2
            for k in ['T', 'P']:
                assert k in list(aliases.values())
        else:
            assert len(aliases.keys()) == 0


class DummyProg1(Prognostic):
    input_properties = {'temperature': {'alias': 'T'}}
    tendency_properties = {'temperature': {'alias': 'TEMP'}}

    def __init__(self):
        self._a = 1

    def __call__(self, state):
        return self._a


class DummyProg2(Prognostic):
    input_properties = {'temperature': {'alias': 't'}}

    def __init__(self):
        self._a = 1

    def __call__(self, state):
        return self._a


class DummyProg3(Prognostic):
    input_properties = {'temperature': {}}
    diagnostic_properties = {'pressure': {}}
    tendency_properties = {'temperature': {}}

    def __init__(self):
        self._a = 1

    def __call__(self, state):
        return self._a


def test_get_component_aliases_with_different_values():
    # two different aliases in the same Component:
    aliases = get_component_aliases(DummyProg1())
    assert len(aliases.keys()) == 1
    assert aliases['temperature'] == 'TEMP'
    # two different aliases in different Components:
    aliases = get_component_aliases(DummyProg1(), DummyProg2())
    assert len(aliases.keys()) == 1
    assert aliases['temperature'] == 't'
    # NO aliases in component
    aliases = get_component_aliases(DummyProg3)
    assert len(aliases.keys()) == 0


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


class CombineDimsTests(unittest.TestCase):

    def test_same_dims(self):
        assert combine_dims(['dim1'], ['dim1']) == ['dim1']

    def test_one_wildcard(self):
        assert combine_dims(['*'], ['dim1']) == ['dim1']

    def test_both_wildcard(self):
        assert combine_dims(['*'], ['*']) == ['*']

    def test_both_wildcard_2d(self):
        assert set(combine_dims(['*', 'dim1'], ['*', 'dim1'])) == {'*', 'dim1'}

    def test_one_wildcard_2d(self):
        assert set(combine_dims(['*', 'dim2'], ['dim1', 'dim2'])) == {'dim1', 'dim2'}

    def test_different_dims(self):
        with self.assertRaises(InvalidPropertyDictError):
            combine_dims(['dim1'], ['dim2'])

    def test_swapped_dims(self):
        result = set(combine_dims(['dim1', 'dim2'], ['dim2', 'dim1']))
        assert result == {'dim1', 'dim2'}

    def test_swapped_dims_with_wildcard(self):
        result = combine_dims(['*', 'dim1', 'dim2'], ['*', 'dim2', 'dim1'])
        assert set(result) == {'*', 'dim1', 'dim2'}


if __name__ == '__main__':
    pytest.main([__file__])
