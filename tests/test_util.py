import unittest
import numpy as np
import pytest
from sympl import (
    TendencyComponent, ensure_no_shared_keys, SharedKeyError, DataArray,
    Stepper, DiagnosticComponent,
    InvalidPropertyDictError)
from sympl._core.util import update_dict_by_adding_another, \
    get_component_aliases
from sympl._core.combine_properties import combine_dims


def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class PrognosticPropertiesContainer(object):

    def __init__(self, input_properties, tendency_properties, diagnostic_properties):
        self.input_properties = input_properties
        self.tendency_properties = tendency_properties
        self.diagnostic_properties = diagnostic_properties


class ImplicitPropertiesContainer(object):

    def __init__(self, input_properties, diagnostic_properties, output_properties):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self.output_properties = output_properties


class DiagnosticPropertiesContainer(object):

    def __init__(self, input_properties, diagnostic_properties):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties


def test_update_dict_by_adding_another_works_on_different_dim_orders():
    dict1 = {
        'quantity': DataArray(
            np.ones([2, 3, 4]),
            dims=['dim1', 'dim2', 'dim3'],
            attrs={'units': 'm'},
        )
    }
    dict2 = {
        'quantity': DataArray(
            np.ones([3, 2, 4]),
            dims=['dim2', 'dim1', 'dim3'],
            attrs={'units': 'm'},
        )
    }
    update_dict_by_adding_another(dict1, dict2)
    assert dict1['quantity'].shape == (2, 3, 4)
    assert np.all(dict1['quantity'].values == 2.)


def test_update_dict_by_adding_another_broadcasts_added_dim():
    dict1 = {
        'quantity': DataArray(
            np.ones([2, 3, 4]),
            dims=['dim1', 'dim2', 'dim3'],
            attrs={'units': 'm'},
        )
    }
    dict2 = {
        'quantity': DataArray(
            np.ones([3, 2]),
            dims=['dim2', 'dim1'],
            attrs={'units': 'm'},
        )
    }
    update_dict_by_adding_another(dict1, dict2)
    assert dict1['quantity'].shape == (2, 3, 4)
    assert np.all(dict1['quantity'].values == 2.)


def test_update_dict_by_adding_another_broadcasts_initial_dim():
    dict1 = {
        'quantity': DataArray(
            np.ones([2, 3]),
            dims=['dim1', 'dim2'],
            attrs={'units': 'm'},
        )
    }
    dict2 = {
        'quantity': DataArray(
            np.ones([3, 2, 4]),
            dims=['dim2', 'dim1', 'dim3'],
            attrs={'units': 'm'},
        )
    }
    update_dict_by_adding_another(dict1, dict2)
    assert dict1['quantity'].shape == (2, 3, 4)
    assert np.all(dict1['quantity'].values == 2.)


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


class DummyTendencyComponent(TendencyComponent):
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


def test_get_component_aliases_prognostic():
    aliases = get_component_aliases(
        PrognosticPropertiesContainer(
            input_properties={
                'temperature': {
                    'alias': 'T',
                }
            },
            tendency_properties={
                'wind': {
                    'alias': 'u'
                }
            },
            diagnostic_properties={
                'specific_humidity': {
                    'alias': 'q'
                },
            }
        )
    )
    assert aliases == {'temperature': 'T', 'wind': 'u', 'specific_humidity': 'q'}


def test_get_component_aliases_no_alias_prognostic():
    aliases = get_component_aliases(
        PrognosticPropertiesContainer(
            input_properties={
                'temperature': {
                }
            },
            tendency_properties={
                'wind': {
                }
            },
            diagnostic_properties={
                'specific_humidity': {
                },
            }
        )
    )
    assert aliases == {}


def test_get_component_aliases_empty_prognostic():
    aliases = get_component_aliases(
        PrognosticPropertiesContainer(
            input_properties={},
            tendency_properties={},
            diagnostic_properties={}
        )
    )
    assert aliases == {}


def test_get_component_aliases_diagnostic():
    aliases = get_component_aliases(
        DiagnosticPropertiesContainer(
            input_properties={
                'temperature': {
                    'alias': 'T',
                }
            },
            diagnostic_properties={
                'specific_humidity': {
                    'alias': 'q'
                },
            }
        )
    )
    assert aliases == {'temperature': 'T', 'specific_humidity': 'q'}


def test_get_component_aliases_no_alias_diagnostic():
    aliases = get_component_aliases(
        DiagnosticPropertiesContainer(
            input_properties={
                'temperature': {
                }
            },
            diagnostic_properties={
                'specific_humidity': {
                },
            }
        )
    )
    assert aliases == {}


def test_get_component_aliases_empty_diagnostic():
    aliases = get_component_aliases(
        DiagnosticPropertiesContainer(
            input_properties={},
            diagnostic_properties={}
        )
    )
    assert aliases == {}


def test_get_component_aliases_implicit():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
                'temperature': {
                    'alias': 'T',
                },
                'input2': {
                    'alias': 'in2',
                }
            },
            output_properties={
                'wind': {
                    'alias': 'u'
                }
            },
            diagnostic_properties={
                'specific_humidity': {
                    'alias': 'q'
                },
            }
        )
    )
    assert aliases == {
        'temperature': 'T', 'input2': 'in2', 'wind': 'u', 'specific_humidity': 'q'}


def test_get_component_aliases_no_alias_implicit():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
                'temperature': {},
            },
            output_properties={
                'wind': {}
            },
            diagnostic_properties={
                'specific_humidity': {},
            }
        )
    )
    assert aliases == {}


def test_get_component_aliases_input_over_diagnostic():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
                'temperature': {'alias': 'correct'},
            },
            output_properties={},
            diagnostic_properties={
                'temperature': {'alias': 'incorrect'},
            }
        )
    )
    assert aliases == {'temperature': 'correct'}


def test_get_component_aliases_input_over_output():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
                'temperature': {'alias': 'correct'},
            },
            output_properties={
                'temperature': {'alias': 'incorrect'},
            },
            diagnostic_properties={}
        )
    )
    assert aliases == {'temperature': 'correct'}


def test_get_component_aliases_output_over_diagnostic():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
            },
            output_properties={
                'temperature': {'alias': 'correct'},
            },
            diagnostic_properties={
                'temperature': {'alias': 'not'},
            }
        )
    )
    assert aliases == {'temperature': 'correct'}


def test_get_component_aliases_diagnostic_over_tendency():
    aliases = get_component_aliases(
        PrognosticPropertiesContainer(
            input_properties={
            },
            tendency_properties={
                'temperature': {'alias': 'not'},
            },
            diagnostic_properties={
                'temperature': {'alias': 'correct'},
            }
        )
    )
    assert aliases == {'temperature': 'correct'}


def test_get_component_aliases_input_over_tendency():
    aliases = get_component_aliases(
        PrognosticPropertiesContainer(
            input_properties={
                'temperature': {'alias': 'T'},
            },
            tendency_properties={
                'temperature': {'alias': 'not'},
            },
            diagnostic_properties={
            }
        )
    )
    assert aliases == {'temperature': 'T'}


def test_get_component_aliases_empty_implicit():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={},
            output_properties={},
            diagnostic_properties={}
        )
    )
    assert aliases == {}


def test_get_component_aliases_all_types_no_overlap():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
                'input1': {
                    'alias': 'in1',
                }
            },
            output_properties={
                'output1': {
                    'alias': 'out1'
                }
            },
            diagnostic_properties={
                'diagnostic1': {
                    'alias': 'diag1'
                },
            }
        ),
        DiagnosticPropertiesContainer(
            input_properties={
                'input2': {
                    'alias': 'in2',
                }
            },
            diagnostic_properties={
                'diagnostic2': {
                    'alias': 'diag2'
                },
            }
        ),
        PrognosticPropertiesContainer(
            input_properties={
                'input3': {
                    'alias': 'in3',
                }
            },
            tendency_properties={
                'tendency3': {
                    'alias': 'tend3'
                }
            },
            diagnostic_properties={
                'diagnostic3': {
                    'alias': 'diag3'
                },
            }
        )
    )
    assert aliases == {
        'input1': 'in1', 'output1': 'out1', 'diagnostic1': 'diag1',
        'input2': 'in2', 'diagnostic2': 'diag2', 'input3': 'in3',
        'tendency3': 'tend3', 'diagnostic3': 'diag3'}

def test_get_component_aliases_all_types_with_overlap():
    aliases = get_component_aliases(
        ImplicitPropertiesContainer(
            input_properties={
                'input1': {
                    'alias': 'in1',
                }
            },
            output_properties={
                'output1': {
                    'alias': 'out1'
                }
            },
            diagnostic_properties={
                'diagnostic1': {
                    'alias': 'diag1'
                },
            }
        ),
        DiagnosticPropertiesContainer(
            input_properties={
                'input1': {
                    'alias': 'in1',
                }
            },
            diagnostic_properties={
                'diagnostic1': {
                    'alias': 'diag1'
                },
            }
        ),
        PrognosticPropertiesContainer(
            input_properties={
                'input1': {
                    'alias': 'in1',
                }
            },
            tendency_properties={
                'tendency1': {
                    'alias': 'tend1',
                }
            },
            diagnostic_properties={
                'diagnostic1': {
                    'alias': 'diag1'
                },
            }
        )
    )
    assert aliases == {
        'input1': 'in1', 'output1': 'out1', 'diagnostic1': 'diag1', 'tendency1': 'tend1'}


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
