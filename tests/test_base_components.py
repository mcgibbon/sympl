import pytest
import mock
import numpy as np
import unittest
from sympl import (
    Prognostic, Diagnostic, Monitor, Implicit, ImplicitPrognostic,
    datetime, timedelta, DataArray, InvalidPropertyDictError,
    ComponentMissingOutputError, ComponentExtraOutputError,
    InvalidStateError
)
import warnings

def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockPrognostic(Prognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self.tendency_properties = tendency_properties
        self.diagnostic_output = diagnostic_output
        self.tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        super(MockPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self.tendency_output, self.diagnostic_output


class MockImplicitPrognostic(ImplicitPrognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self.tendency_properties = tendency_properties
        self.diagnostic_output = diagnostic_output
        self.tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicitPrognostic, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.tendency_output, self.diagnostic_output


class MockDiagnostic(Diagnostic):

    input_properties = None
    diagnostic_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, diagnostic_output,
            **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self.diagnostic_output = diagnostic_output
        self.times_called = 0
        self.state_given = None
        super(MockDiagnostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self.diagnostic_output


class MockImplicit(Implicit):

    input_properties = None
    diagnostic_properties = None
    output_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, output_properties,
            diagnostic_output, state_output,
            **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self.output_properties = output_properties
        self.diagnostic_output = diagnostic_output
        self.state_output = state_output
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicit, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.diagnostic_output, self.state_output


class MockMonitor(Monitor):

    def store(self, state):
        return

class BadMockPrognostic(Prognostic):

    input_properties = {}
    tendency_properties = {}
    diagnostic_properties = {}

    def __init__(self):
        pass

    def array_call(self, state):
        return {}, {}


class BadMockImplicitPrognostic(ImplicitPrognostic):

    input_properties = {}
    tendency_properties = {}
    diagnostic_properties = {}

    def __init__(self):
        pass

    def array_call(self, state, timestep):
        return {}, {}


class BadMockDiagnostic(Diagnostic):

    input_properties = {}
    diagnostic_properties = {}

    def __init__(self):
        pass

    def array_call(self, state):
        return {}


class BadMockImplicit(Implicit):

    input_properties = {}
    diagnostic_properties = {}
    output_properties = {}

    def __init__(self):
        pass

    def array_call(self, state, timestep):
        return {}, {}


class InputTestBase():

    def test_cannot_overlap_input_aliases(self):
        input_properties = {
            'input1': {'dims': ['dim1'], 'units': 'm', 'alias': 'input'},
            'input2': {'dims': ['dim1'], 'units': 'm', 'alias': 'input'}
        }
        with self.assertRaises(InvalidPropertyDictError):
            self.get_component(input_properties=input_properties)

    def test_raises_when_input_missing(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {'time': timedelta(0)}
        with self.assertRaises(InvalidStateError):
            self.call_component(component, state)

    def test_raises_when_input_incorrect_units(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.zeros([10]),
                dims=['dim1'],
                attrs={'units': 's'},
            ),
        }
        with self.assertRaises(InvalidStateError):
            self.call_component(component, state)

    def test_raises_when_input_incorrect_dims(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.zeros([10]),
                dims=['dim2'],
                attrs={'units': 'm'},
            ),
        }
        with self.assertRaises(InvalidStateError):
            self.call_component(component, state)

    def test_raises_when_input_conflicting_dim_lengths(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
            'input1': {
                'dims': ['dim2'],
                'units': 'm',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.zeros([10]),
                dims=['dim1'],
                attrs={'units': 'm'},
            ),
            'input2': DataArray(
                np.zeros([7]),
                dims=['dim1'],
                attrs={'units': 'm'},
            ),
        }
        with self.assertRaises(InvalidStateError):
            self.call_component(component, state)

    def test_collects_independent_wildcard_dims(self):
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 'm',
            },
            'input2': {
                'dims': ['*'],
                'units': 'm',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.zeros([4]),
                dims=['dim1'],
                attrs={'units': 'm'},
            ),
            'input2': DataArray(
                np.zeros([3]),
                dims=['dim2'],
                attrs={'units': 'm'},
            ),
        }
        self.call_component(component, state)
        given = component.state_given
        assert len(given.keys()) == 3
        assert 'time' in given.keys()
        assert 'input1' in given.keys()
        assert given['input1'].shape == (12,)
        assert 'input2' in given.keys()
        assert given['input2'].shape == (12,)

    def test_accepts_when_input_swapped_dims(self):
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 'm',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.zeros([3, 4]),
                dims=['dim2', 'dim1'],
                attrs={'units': 'm'},
            ),
        }
        self.call_component(component, state)
        assert component.state_given['input1'].shape == (4, 3)

    def test_input_requires_dims(self):
        input_properties = {'input1': {'units': 'm'}}
        with self.assertRaises(InvalidPropertyDictError):
            self.get_component(input_properties=input_properties)

    def test_input_requires_units(self):
        input_properties = {'input1': {'dims': ['dim1']}}
        with self.assertRaises(InvalidPropertyDictError):
            self.get_component(input_properties=input_properties)

    def test_input_no_transformations(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        self.call_component(component, state)
        assert len(component.state_given) == 2
        assert 'time' in component.state_given.keys()
        assert 'input1' in component.state_given.keys()
        assert isinstance(component.state_given['input1'], np.ndarray)
        assert np.all(component.state_given['input1'] == np.ones([10]))

    def test_input_converts_units(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'km'}
            )
        }
        self.call_component(component, state)
        assert len(component.state_given) == 2
        assert 'time' in component.state_given.keys()
        assert 'input1' in component.state_given.keys()
        assert isinstance(component.state_given['input1'], np.ndarray)
        assert np.all(component.state_given['input1'] == np.ones([10])*1000.)

    def test_input_converts_temperature_units(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'degK'
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'degC'}
            )
        }
        self.call_component(component, state)
        assert len(component.state_given) == 2
        assert 'time' in component.state_given.keys()
        assert 'input1' in component.state_given.keys()
        assert isinstance(component.state_given['input1'], np.ndarray)
        assert np.all(component.state_given['input1'] == np.ones([10])*274.15)

    def test_input_collects_one_dimension(self):
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 'm'
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        self.call_component(component, state)
        assert len(component.state_given) == 2
        assert 'time' in component.state_given.keys()
        assert 'input1' in component.state_given.keys()
        assert isinstance(component.state_given['input1'], np.ndarray)
        assert np.all(component.state_given['input1'] == np.ones([10]))

    def test_input_is_aliased(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'in1',
            }
        }
        component = self.get_component(input_properties=input_properties)
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        self.call_component(component, state)
        assert len(component.state_given) == 2
        assert 'time' in component.state_given.keys()
        assert 'in1' in component.state_given.keys()
        assert isinstance(component.state_given['in1'], np.ndarray)
        assert np.all(component.state_given['in1'] == np.ones([10]))


class DiagnosticTestBase():

    def test_diagnostic_requires_dims(self):
        diagnostic_properties = {'diag1': {'units': 'm'}}
        with self.assertRaises(InvalidPropertyDictError):
            self.get_component(diagnostic_properties=diagnostic_properties)

    def test_diagnostic_requires_units(self):
        diagnostic_properties = {'diag1': {'dims': ['dim1']}}
        with self.assertRaises(InvalidPropertyDictError):
            self.get_component(diagnostic_properties=diagnostic_properties)

    def test_diagnostic_warns_when_units_incompatible_with_input(self):
        input_properties = {
            'diag1': {'units': 'km', 'dims': ['dim1', 'dim2']}
        }
        diagnostic_properties = {
            'diag1': {'units': 'seconds', 'dims': ['dim1', 'dim2']}
        }
        with warnings.catch_warnings(record=True) as w:
            self.get_component(
                input_properties=input_properties,
                diagnostic_properties=diagnostic_properties
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'units' in str(w[-1].message)
            assert 'diag1' in str(w[-1].message)
            assert 'seconds' in str(w[-1].message)
            assert 'km' in str(w[-1].message)

    def test_diagnostic_requires_correct_number_of_dims(self):
        input_properties = {
            'input1': {'units': 'm', 'dims': ['dim1', 'dim2']}
        }
        diagnostic_properties = {
            'diag1': {'units': 'm', 'dims': ['dim1', 'dim2']}
        }
        diagnostic_output = {'diag1': np.zeros([10]),}
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10, 2]),
                dims=['dim1', 'dim2'],
                attrs={'units': 'm'}
            )
        }
        component = self.get_component(
            input_properties = input_properties,
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output,
        )
        with self.assertRaises(InvalidPropertyDictError):
            _, _ = self.call_component(component, state)

    def test_diagnostic_requires_correct_dim_length(self):
        input_properties = {
            'input1': {'units': 'm', 'dims': ['dim1', 'dim2']}
        }
        diagnostic_properties = {
            'diag1': {'units': 'm', 'dims': ['dim1', 'dim2']}
        }
        diagnostic_output = {'diag1': np.zeros([5, 2]),}
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10, 2]),
                dims=['dim1', 'dim2'],
                attrs={'units': 'm'}
            )
        }
        component = self.get_component(
            input_properties=input_properties,
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output
        )
        with self.assertRaises(InvalidPropertyDictError):
            _, _ = self.call_component(component, state)

    def test_diagnostic_uses_input_dims(self):
        input_properties = {'diag1': {'dims': ['dim1'], 'units': 'm'}}
        diagnostic_properties = {'diag1': {'units': 'm'}}
        self.get_component(
            input_properties=input_properties,
            diagnostic_properties=diagnostic_properties
        )

    def test_diagnostic_doesnt_use_input_units(self):
        input_properties = {'diag1': {'dims': ['dim1'], 'units': 'm'}}
        diagnostic_properties = {'diag1': {'dims': ['dim1']}}
        with self.assertRaises(InvalidPropertyDictError):
            self.get_component(
                input_properties=input_properties,
                diagnostic_properties=diagnostic_properties
            )

    def test_diagnostics_no_transformations(self):
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        component = self.get_component(
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output,
        )
        state = {'time': timedelta(0)}
        diagnostics = self.get_diagnostics(self.call_component(component, state))
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_restoring_dims(self):
        input_properties = {
            'input1': {
                'dims': ['*', 'dim1'],
                'units': 'm',
            }
        }
        diagnostic_properties = {
            'output1': {
                'dims': ['*', 'dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {
            'output1': np.ones([1, 10]),
        }
        component = self.get_component(
            input_properties=input_properties,
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output,
        )
        state = {
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}),
            'time': timedelta(0)}
        diagnostics = self.get_diagnostics(self.call_component(component, state))
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_with_alias(self):
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'out1',
            }
        }
        diagnostic_output = {
            'out1': np.ones([10]),
        }
        component = self.get_component(
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output,
        )
        state = {'time': timedelta(0)}
        diagnostics = self.get_diagnostics(self.call_component(component, state))
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_with_alias_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'out1',
            }
        }
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        diagnostic_output = {
            'out1': np.ones([10]),
        }
        component = self.get_component(
            input_properties=input_properties,
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output,
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        diagnostics = self.get_diagnostics(self.call_component(component, state))
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_with_dims_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        diagnostic_properties = {
            'output1': {
                'units': 'm',
            }
        }
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        component = self.get_component(
            input_properties=input_properties,
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output,
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        diagnostics = self.get_diagnostics(self.call_component(component, state))
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_raises_when_diagnostic_not_given(self):
        diagnostic_properties = {
            'diag1': {
                'dims': ['dims1'],
                'units': 'm',
            }
        }
        diagnostic_output = {}
        diagnostic = self.get_component(
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentMissingOutputError):
            self.call_component(diagnostic, state)

    def test_raises_when_extraneous_diagnostic_given(self):
        diagnostic_properties = {}
        diagnostic_output = {
            'diag1': np.zeros([10])
        }
        diagnostic = self.get_component(
            diagnostic_properties=diagnostic_properties,
            diagnostic_output=diagnostic_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentExtraOutputError):
            self.call_component(diagnostic, state)


class PrognosticTests(unittest.TestCase, InputTestBase):

    component_class = MockPrognostic

    def call_component(self, component, state):
        return component(state)

    def get_component(
            self, input_properties=None, tendency_properties=None,
            diagnostic_properties=None, tendency_output=None,
            diagnostic_output=None):
        return MockPrognostic(
            input_properties=input_properties or {},
            tendency_properties=tendency_properties or {},
            diagnostic_properties=diagnostic_properties or {},
            tendency_output=tendency_output or {},
            diagnostic_output=diagnostic_output or {},
        )

    def get_diagnostics(self, result):
        return result[1]

    def test_cannot_use_bad_component(self):
        component = BadMockPrognostic()
        with self.assertRaises(RuntimeError):
            self.call_component(component, {'time': timedelta(0)})

    def test_subclass_check(self):
        class MyPrognostic(object):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __call__(self):
                pass
            def array_call(self):
                pass

        instance = MyPrognostic()
        assert isinstance(instance, Prognostic)

    def test_tendency_warns_when_units_incompatible_with_input(self):
        input_properties = {
            'input1': {'units': 'km', 'dims': ['dim1', 'dim2']}
        }
        tendency_properties = {
            'input1': {'units': 'degK/s', 'dims': ['dim1', 'dim2']}
        }
        with warnings.catch_warnings(record=True) as w:
            self.get_component(
                input_properties=input_properties,
                tendency_properties=tendency_properties
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'units' in str(w[-1].message)
            assert 'input1' in str(w[-1].message)
            assert 'degK/s' in str(w[-1].message)
            assert 'km' in str(w[-1].message)

    def test_two_components_are_not_instances_of_each_other(self):
        class MyPrognostic1(Prognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def array_call(self, state):
                pass

        class MyPrognostic2(Prognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def array_call(self, state):
                pass

        prog1 = MyPrognostic1()
        prog2 = MyPrognostic2()
        assert not isinstance(prog1, MyPrognostic2)
        assert not isinstance(prog2, MyPrognostic1)

    def test_ducktype_not_instance_of_subclass(self):
        class MyPrognostic1(object):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __init__(self):
                pass
            def array_call(self, state):
                pass

        class MyPrognostic2(Prognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def array_call(self, state):
                pass

        prog1 = MyPrognostic1()
        assert not isinstance(prog1, MyPrognostic2)

    def test_empty_prognostic(self):
        prognostic = self.component_class({}, {}, {}, {}, {})
        tendencies, diagnostics = self.call_component(
            prognostic, {'time': timedelta(seconds=0)})
        assert tendencies == {}
        assert diagnostics == {}
        assert len(prognostic.state_given) == 1
        assert 'time' in prognostic.state_given.keys()
        assert prognostic.state_given['time'] == timedelta(seconds=0)
        assert prognostic.times_called == 1

    def test_tendency_requires_dims(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {'tend1': {'units': 'm'}}
        diagnostic_output = {}
        tendency_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                tendency_properties,
                diagnostic_output, tendency_output
            )

    def test_tendency_uses_base_dims(self):
        input_properties = {'diag1': {'dims': ['dim1'], 'units': 'm'}}
        diagnostic_properties = {}
        tendency_properties = {'diag1': {'units': 'm'}}
        diagnostic_output = {}
        tendency_output = {}
        self.component_class(
            input_properties, diagnostic_properties,
            tendency_properties,
            diagnostic_output, tendency_output
        )

    def test_tendency_doesnt_use_base_units(self):
        input_properties = {'diag1': {'dims': ['dim1'], 'units': 'm'}}
        diagnostic_properties = {}
        tendency_properties = {'diag1': {'dims': ['dim1']}}
        diagnostic_output = {}
        tendency_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                tendency_properties,
                diagnostic_output, tendency_output
            )

    def test_tendency_requires_units(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {'tend1': {'dims': ['dim1']}}
        diagnostic_output = {}
        tendency_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                tendency_properties,
                diagnostic_output, tendency_output
            )

    def test_raises_when_tendency_not_given(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'tend1': {
                'dims': ['dims1'],
                'units': 'm',
            }
        }
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentMissingOutputError):
            _, _ = self.call_component(prognostic, state)

    def test_cannot_overlap_input_aliases(self):
        input_properties = {
            'input1': {'dims': ['dim1'], 'units': 'm', 'alias': 'input'},
            'input2': {'dims': ['dim1'], 'units': 'm', 'alias': 'input'}
        }
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                tendency_properties,
                diagnostic_output, tendency_output
            )

    def test_cannot_overlap_diagnostic_aliases(self):
        input_properties = {
        }
        diagnostic_properties = {
            'diag1': {'dims': ['dim1'], 'units': 'm', 'alias': 'diag'},
            'diag2': {'dims': ['dim1'], 'units': 'm', 'alias': 'diag'}
        }
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                tendency_properties,
                diagnostic_output, tendency_output
            )

    def test_cannot_overlap_tendency_aliases(self):
        input_properties = {
        }
        diagnostic_properties = {
        }
        tendency_properties = {
            'tend1': {'dims': ['dim1'], 'units': 'm', 'alias': 'tend'},
            'tend2': {'dims': ['dim1'], 'units': 'm', 'alias': 'tend'}
        }
        diagnostic_output = {}
        tendency_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                tendency_properties,
                diagnostic_output, tendency_output
            )

    def test_raises_when_extraneous_tendency_given(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {
            'tend1': np.zeros([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentExtraOutputError):
            _, _ = self.call_component(prognostic, state)

    def test_raises_when_diagnostic_not_given(self):
        input_properties = {}
        diagnostic_properties = {
            'diag1': {
                'dims': ['dims1'],
                'units': 'm',
            }
        }
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentMissingOutputError):
            _, _ = self.call_component(prognostic, state)

    def test_raises_when_extraneous_diagnostic_given(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {
            'diag1': np.zeros([10])
        }
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentExtraOutputError):
            _, _ = self.call_component(prognostic, state)

    def test_tendencies_no_transformations(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            }}
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        tendencies, _ = self.call_component(prognostic, state)
        assert len(tendencies) == 1
        assert 'output1' in tendencies.keys()
        assert isinstance(tendencies['output1'], DataArray)
        assert len(tendencies['output1'].dims) == 1
        assert 'dim1' in tendencies['output1'].dims
        assert 'units' in tendencies['output1'].attrs
        assert tendencies['output1'].attrs['units'] == 'm/s'
        assert np.all(tendencies['output1'].values == np.ones([10]))

    def test_tendencies_with_alias(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s',
                'alias': 'out1',
            }}
        diagnostic_output = {}
        tendency_output = {
            'out1': np.ones([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        tendencies, _ = self.call_component(prognostic, state)
        assert len(tendencies) == 1
        assert 'output1' in tendencies.keys()
        assert isinstance(tendencies['output1'], DataArray)
        assert len(tendencies['output1'].dims) == 1
        assert 'dim1' in tendencies['output1'].dims
        assert 'units' in tendencies['output1'].attrs
        assert tendencies['output1'].attrs['units'] == 'm/s'
        assert np.all(tendencies['output1'].values == np.ones([10]))

    def test_tendencies_with_alias_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'out1',
            }
        }
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        }
        diagnostic_output = {}
        tendency_output = {
            'out1': np.ones([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        tendencies, _ = self.call_component(prognostic, state)
        assert len(tendencies) == 1
        assert 'output1' in tendencies.keys()
        assert isinstance(tendencies['output1'], DataArray)
        assert len(tendencies['output1'].dims) == 1
        assert 'dim1' in tendencies['output1'].dims
        assert 'units' in tendencies['output1'].attrs
        assert tendencies['output1'].attrs['units'] == 'm/s'
        assert np.all(tendencies['output1'].values == np.ones([10]))

    def test_tendencies_with_dims_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'units': 'm/s',
            }
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        tendencies, _ = self.call_component(prognostic, state)
        assert len(tendencies) == 1
        assert 'output1' in tendencies.keys()
        assert isinstance(tendencies['output1'], DataArray)
        assert len(tendencies['output1'].dims) == 1
        assert 'dim1' in tendencies['output1'].dims
        assert 'units' in tendencies['output1'].attrs
        assert tendencies['output1'].attrs['units'] == 'm/s'
        assert np.all(tendencies['output1'].values == np.ones([10]))

    def test_tendencies_in_diagnostics_no_tendency(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, tendencies_in_diagnostics=True
        )
        assert prognostic.input_properties == {}
        assert prognostic.diagnostic_properties == {}
        assert prognostic.tendency_properties == {}
        state = {'time': timedelta(0)}
        _, diagnostics = self.call_component(prognostic, state)
        assert diagnostics == {}

    def test_tendencies_in_diagnostics_one_tendency(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            }
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 20.,
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, tendencies_in_diagnostics=True,
        )
        tendency_name = 'output1_tendency_from_{}'.format(prognostic.__class__.__name__)
        assert len(prognostic.diagnostic_properties) == 1
        assert tendency_name in prognostic.diagnostic_properties.keys()
        properties = prognostic.diagnostic_properties[tendency_name]
        assert properties['dims'] == ['dim1']
        assert properties['units'] == 'm/s'
        state = {
            'time': timedelta(0),
        }
        _, diagnostics = self.call_component(prognostic, state)
        assert tendency_name in diagnostics.keys()
        assert len(
            diagnostics[tendency_name].dims) == 1
        assert 'dim1' in diagnostics[tendency_name].dims
        assert diagnostics[tendency_name].attrs['units'] == 'm/s'
        assert np.all(diagnostics[tendency_name].values == 20.)

    def test_tendencies_in_diagnostics_one_tendency_dims_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'units': 'm/s'
            }
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 20.,
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, tendencies_in_diagnostics=True,
        )
        tendency_name = 'output1_tendency_from_{}'.format(prognostic.__class__.__name__)
        assert len(prognostic.diagnostic_properties) == 1
        assert tendency_name in prognostic.diagnostic_properties.keys()
        properties = prognostic.diagnostic_properties[tendency_name]
        assert properties['dims'] == ['dim1']
        assert properties['units'] == 'm/s'
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}),
        }
        _, diagnostics = self.call_component(prognostic, state)
        assert tendency_name in diagnostics.keys()
        assert len(
            diagnostics[tendency_name].dims) == 1
        assert 'dim1' in diagnostics[tendency_name].dims
        assert diagnostics[tendency_name].attrs['units'] == 'm/s'
        assert np.all(diagnostics[tendency_name].values == 20.)

    def test_tendencies_in_diagnostics_one_tendency_with_component_name(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            }
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 20.,
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, tendencies_in_diagnostics=True,
            name='component',
        )
        tendency_name = 'output1_tendency_from_component'
        assert len(prognostic.diagnostic_properties) == 1
        assert tendency_name in prognostic.diagnostic_properties.keys()
        properties = prognostic.diagnostic_properties[tendency_name]
        assert properties['dims'] == ['dim1']
        assert properties['units'] == 'm/s'
        state = {
            'time': timedelta(0),
        }
        _, diagnostics = self.call_component(prognostic, state)
        print(diagnostics.keys())
        assert tendency_name in diagnostics.keys()
        assert len(
            diagnostics[tendency_name].dims) == 1
        assert 'dim1' in diagnostics[tendency_name].dims
        assert diagnostics[tendency_name].attrs['units'] == 'm/s'
        assert np.all(diagnostics[tendency_name].values == 20.)


class ImplicitPrognosticTests(PrognosticTests):

    component_class = MockImplicitPrognostic

    def call_component(self, component, state):
        return component(state, timedelta(seconds=1))

    def get_component(
            self, input_properties=None, tendency_properties=None,
            diagnostic_properties=None, tendency_output=None,
            diagnostic_output=None):
        return MockImplicitPrognostic(
            input_properties=input_properties or {},
            tendency_properties=tendency_properties or {},
            diagnostic_properties=diagnostic_properties or {},
            tendency_output=tendency_output or {},
            diagnostic_output=diagnostic_output or {},
        )

    def test_cannot_use_bad_component(self):
        component = BadMockImplicitPrognostic()
        with self.assertRaises(RuntimeError):
            self.call_component(component, {'time': timedelta(0)})

    def test_subclass_check(self):
        class MyImplicitPrognostic(object):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __call__(self, state, timestep):
                pass
            def array_call(self, state, timestep):
                pass

        instance = MyImplicitPrognostic()
        assert isinstance(instance, ImplicitPrognostic)

    def test_two_components_are_not_instances_of_each_other(self):
        class MyImplicitPrognostic1(ImplicitPrognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''

            def array_call(self, state, timestep):
                pass

        class MyImplicitPrognostic2(ImplicitPrognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''

            def array_call(self, state):
                pass

        prog1 = MyImplicitPrognostic1()
        prog2 = MyImplicitPrognostic2()
        assert not isinstance(prog1, MyImplicitPrognostic2)
        assert not isinstance(prog2, MyImplicitPrognostic1)

    def test_ducktype_not_instance_of_subclass(self):
        class MyImplicitPrognostic1(object):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __init__(self):
                pass
            def array_call(self, state, timestep):
                pass

        class MyImplicitPrognostic2(ImplicitPrognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''

            def array_call(self, state):
                pass

        prog1 = MyImplicitPrognostic1()
        assert not isinstance(prog1, MyImplicitPrognostic2)

    def test_subclass_is_not_prognostic(self):
        class MyImplicitPrognostic(ImplicitPrognostic):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def array_call(self, state, timestep):
                pass

        instance = MyImplicitPrognostic()
        assert not isinstance(instance, Prognostic)

    def test_ducktype_is_not_prognostic(self):
        class MyImplicitPrognostic(object):
            input_properties = {}
            diagnostic_properties = {}
            tendency_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __call__(self, state, timestep):
                pass
            def array_call(self, state, timestep):
                pass

        instance = MyImplicitPrognostic()
        assert not isinstance(instance, Prognostic)

    def test_timedelta_is_passed(self):
        prognostic = MockImplicitPrognostic({}, {}, {}, {}, {})
        tendencies, diagnostics = prognostic(
            {'time': timedelta(seconds=0)}, timedelta(seconds=5))
        assert tendencies == {}
        assert diagnostics == {}
        assert prognostic.timestep_given == timedelta(seconds=5)
        assert prognostic.times_called == 1


class DiagnosticTests(unittest.TestCase, InputTestBase, DiagnosticTestBase):

    component_class = MockDiagnostic

    def call_component(self, component, state):
        return component(state)

    def get_component(
            self, input_properties=None,
            diagnostic_properties=None,
            diagnostic_output=None):
        return MockDiagnostic(
            input_properties=input_properties or {},
            diagnostic_properties=diagnostic_properties or {},
            diagnostic_output=diagnostic_output or {},
        )

    def get_diagnostics(self, result):
        return result

    def test_cannot_use_bad_component(self):
        component = BadMockDiagnostic()
        with self.assertRaises(RuntimeError):
            self.call_component(component, {'time': timedelta(0)})

    def test_subclass_check(self):
        class MyDiagnostic(object):
            input_properties = {}
            diagnostic_properties = {}
            def __call__(self, state):
                pass
            def array_call(self, state):
                pass

        instance = MyDiagnostic()
        assert isinstance(instance, Diagnostic)

    def test_two_components_are_not_instances_of_each_other(self):
        class MyDiagnostic1(Diagnostic):
            input_properties = {}
            diagnostic_properties = {}

            def array_call(self, state):
                pass

        class MyDiagnostic2(Diagnostic):
            input_properties = {}
            diagnostic_properties = {}

            def array_call(self, state):
                pass

        diag1 = MyDiagnostic1()
        diag2 = MyDiagnostic2()
        assert not isinstance(diag1, MyDiagnostic2)
        assert not isinstance(diag2, MyDiagnostic1)

    def test_ducktype_not_instance_of_subclass(self):
        class MyDiagnostic1(object):
            input_properties = {}
            diagnostic_properties = {}
            def __init__(self):
                pass
            def array_call(self, state):
                pass

        class MyDiagnostic2(Diagnostic):
            input_properties = {}
            diagnostic_properties = {}

            def array_call(self, state):
                pass

        diag1 = MyDiagnostic1()
        assert not isinstance(diag1, MyDiagnostic2)

    def test_empty_diagnostic(self):
        diagnostic = self.component_class({}, {}, {})
        diagnostics = diagnostic({'time': timedelta(seconds=0)})
        assert diagnostics == {}
        assert len(diagnostic.state_given) == 1
        assert 'time' in diagnostic.state_given.keys()
        assert diagnostic.state_given['time'] == timedelta(seconds=0)
        assert diagnostic.times_called == 1


class ImplicitTests(unittest.TestCase, InputTestBase, DiagnosticTestBase):

    component_class = MockImplicit

    def call_component(self, component, state):
        return component(state, timedelta(seconds=1))

    def get_component(
            self, input_properties=None, output_properties=None,
            diagnostic_properties=None, state_output=None,
            diagnostic_output=None):
        return MockImplicit(
            input_properties=input_properties or {},
            output_properties=output_properties or {},
            diagnostic_properties=diagnostic_properties or {},
            state_output=state_output or {},
            diagnostic_output=diagnostic_output or {},
        )

    def get_diagnostics(self, result):
        return result[0]

    def test_cannot_use_bad_component(self):
        component = BadMockImplicit()
        with self.assertRaises(RuntimeError):
            self.call_component(component, {'time': timedelta(0)})

    def test_subclass_check(self):
        class MyImplicit(object):
            input_properties = {}
            diagnostic_properties = {}
            output_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __call__(self, state, timestep):
                pass
            def array_call(self, state, timestep):
                pass

        instance = MyImplicit()
        assert isinstance(instance, Implicit)

    def test_output_warns_when_units_incompatible_with_input(self):
        input_properties = {
            'input1': {'units': 'km', 'dims': ['dim1', 'dim2']}
        }
        output_properties = {
            'input1': {'units': 'degK', 'dims': ['dim1', 'dim2']}
        }
        with warnings.catch_warnings(record=True) as w:
            self.get_component(
                input_properties=input_properties,
                output_properties=output_properties,
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'units' in str(w[-1].message)
            assert 'input1' in str(w[-1].message)
            assert 'degK' in str(w[-1].message)
            assert 'km' in str(w[-1].message)

    def test_two_components_are_not_instances_of_each_other(self):
        class MyImplicit1(Implicit):
            input_properties = {}
            diagnostic_properties = {}
            output_properties = {}
            tendencies_in_diagnostics = False
            name = ''

            def array_call(self, state):
                pass

        class MyImplicit2(Implicit):
            input_properties = {}
            diagnostic_properties = {}
            output_properties = {}
            tendencies_in_diagnostics = False
            name = ''

            def array_call(self, state):
                pass

        implicit1 = MyImplicit1()
        implicit2 = MyImplicit2()
        assert not isinstance(implicit1, MyImplicit2)
        assert not isinstance(implicit2, MyImplicit1)

    def test_ducktype_not_instance_of_subclass(self):
        class MyImplicit1(object):
            input_properties = {}
            diagnostic_properties = {}
            output_properties = {}
            tendencies_in_diagnostics = False
            name = ''
            def __init__(self):
                pass
            def array_call(self, state):
                pass

        class MyImplicit2(Implicit):
            input_properties = {}
            diagnostic_properties = {}
            output_properties = {}
            tendencies_in_diagnostics = False
            name = ''

            def array_call(self, state):
                pass

        implicit1 = MyImplicit1()
        assert not isinstance(implicit1, MyImplicit2)

    def test_empty_implicit(self):
        implicit = self.component_class(
            {}, {}, {}, {}, {})
        tendencies, diagnostics = self.call_component(
            implicit, {'time': timedelta(seconds=0)})
        assert tendencies == {}
        assert diagnostics == {}
        assert len(implicit.state_given) == 1
        assert 'time' in implicit.state_given.keys()
        assert implicit.state_given['time'] == timedelta(seconds=0)
        assert implicit.times_called == 1

    def test_output_requires_dims(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {'diag1': {'units': 'm'}}
        diagnostic_output = {}
        state_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                output_properties,
                diagnostic_output, state_output
            )

    def test_output_uses_base_dims(self):
        input_properties = {'diag1': {'dims': ['dim1'], 'units': 'm'}}
        diagnostic_properties = {}
        output_properties = {'diag1': {'units': 'm'}}
        diagnostic_output = {}
        state_output = {}
        self.component_class(
            input_properties, diagnostic_properties,
            output_properties,
            diagnostic_output, state_output
        )

    def test_output_doesnt_use_base_units(self):
        input_properties = {'diag1': {'dims': ['dim1'], 'units': 'm'}}
        diagnostic_properties = {}
        output_properties = {'diag1': {'dims': ['dim1']}}
        diagnostic_output = {}
        state_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                output_properties,
                diagnostic_output, state_output
            )

    def test_output_requires_units(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {'output1': {'dims': ['dim1']}}
        diagnostic_output = {}
        state_output = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                output_properties,
                diagnostic_output, state_output
            )

    def test_cannot_overlap_output_aliases(self):
        input_properties = {
        }
        diagnostic_properties = {
        }
        output_properties = {
            'out1': {'dims': ['dim1'], 'units': 'm', 'alias': 'out'},
            'out2': {'dims': ['dim1'], 'units': 'm', 'alias': 'out'}
        }
        diagnostic_output = {}
        output_state = {}
        with self.assertRaises(InvalidPropertyDictError):
            self.component_class(
                input_properties, diagnostic_properties,
                output_properties,
                diagnostic_output, output_state
            )

    def test_timedelta_is_passed(self):
        implicit = MockImplicit({}, {}, {}, {}, {})
        tendencies, diagnostics = implicit(
            {'time': timedelta(seconds=0)}, timedelta(seconds=5))
        assert tendencies == {}
        assert diagnostics == {}
        assert implicit.timestep_given == timedelta(seconds=5)
        assert implicit.times_called == 1

    def test_raises_when_output_not_given(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dims1'],
                'units': 'm',
            }
        }
        diagnostic_output = {}
        state_output = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, state_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentMissingOutputError):
            _, _ = self.call_component(implicit, state)

    def test_raises_when_extraneous_output_given(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        state_output = {
            'tend1': np.zeros([10]),
        }
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, state_output
        )
        state = {'time': timedelta(0)}
        with self.assertRaises(ComponentExtraOutputError):
            _, _ = self.call_component(implicit, state)

    def test_output_no_transformations(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            }}
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {'time': timedelta(0)}
        _, output = self.call_component(prognostic, state)
        assert len(output) == 1
        assert 'output1' in output.keys()
        assert isinstance(output['output1'], DataArray)
        assert len(output['output1'].dims) == 1
        assert 'dim1' in output['output1'].dims
        assert 'units' in output['output1'].attrs
        assert output['output1'].attrs['units'] == 'm/s'
        assert np.all(output['output1'].values == np.ones([10]))

    def test_output_with_alias(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s',
                'alias': 'out1',
            }}
        diagnostic_output = {}
        output_state = {
            'out1': np.ones([10]),
        }
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {'time': timedelta(0)}
        _, output = self.call_component(implicit, state)
        assert len(output) == 1
        assert 'output1' in output.keys()
        assert isinstance(output['output1'], DataArray)
        assert len(output['output1'].dims) == 1
        assert 'dim1' in output['output1'].dims
        assert 'units' in output['output1'].attrs
        assert output['output1'].attrs['units'] == 'm/s'
        assert np.all(output['output1'].values == np.ones([10]))

    def test_output_with_alias_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'out1',
            }
        }
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        }
        diagnostic_output = {}
        output_state = {
            'out1': np.ones([10]),
        }
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, output = self.call_component(implicit, state)
        assert len(output) == 1
        assert 'output1' in output.keys()
        assert isinstance(output['output1'], DataArray)
        assert len(output['output1'].dims) == 1
        assert 'dim1' in output['output1'].dims
        assert 'units' in output['output1'].attrs
        assert output['output1'].attrs['units'] == 'm/s'
        assert np.all(output['output1'].values == np.ones([10]))

    def test_output_with_dims_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'units': 'm/s',
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]),
        }
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, output = self.call_component(implicit, state)
        assert len(output) == 1
        assert 'output1' in output.keys()
        assert isinstance(output['output1'], DataArray)
        assert len(output['output1'].dims) == 1
        assert 'dim1' in output['output1'].dims
        assert 'units' in output['output1'].attrs
        assert output['output1'].attrs['units'] == 'm/s'
        assert np.all(output['output1'].values == np.ones([10]))

    def test_tendencies_in_diagnostics_no_tendency(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = MockImplicit(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state, tendencies_in_diagnostics=True
        )
        assert implicit.input_properties == {}
        assert implicit.diagnostic_properties == {}
        assert implicit.output_properties == {}
        state = {'time': timedelta(0)}
        diagnostics, _ = implicit(state, timedelta(seconds=5))
        assert diagnostics == {}

    def test_tendencies_in_diagnostics_one_tendency(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]) * 20.,
        }
        implicit = MockImplicit(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state, tendencies_in_diagnostics=True,
        )
        assert len(implicit.diagnostic_properties) == 1
        assert 'output1_tendency_from_mockimplicit' in implicit.diagnostic_properties.keys()
        assert 'output1' in input_properties.keys(), 'Implicit needs original value to calculate tendency'
        assert input_properties['output1']['dims'] == ['dim1']
        assert input_properties['output1']['units'] == 'm'
        properties = implicit.diagnostic_properties[
            'output1_tendency_from_mockimplicit']
        assert properties['dims'] == ['dim1']
        assert properties['units'] == 'm s^-1'
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10])*10.,
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        diagnostics, _ = implicit(state, timedelta(seconds=5))
        assert 'output1_tendency_from_mockimplicit' in diagnostics.keys()
        assert len(
            diagnostics['output1_tendency_from_mockimplicit'].dims) == 1
        assert 'dim1' in diagnostics['output1_tendency_from_mockimplicit'].dims
        assert diagnostics['output1_tendency_from_mockimplicit'].attrs['units'] == 'm s^-1'
        assert np.all(
            diagnostics['output1_tendency_from_mockimplicit'].values == 2.)

    def test_tendencies_in_diagnostics_one_tendency_dims_from_input(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'units': 'm'
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]) * 20.,
        }
        implicit = MockImplicit(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state, tendencies_in_diagnostics=True,
        )
        assert len(implicit.diagnostic_properties) == 1
        assert 'output1_tendency_from_mockimplicit' in implicit.diagnostic_properties.keys()
        assert 'output1' in input_properties.keys(), 'Implicit needs original value to calculate tendency'
        assert input_properties['output1']['dims'] == ['dim1']
        assert input_properties['output1']['units'] == 'm'
        properties = implicit.diagnostic_properties[
            'output1_tendency_from_mockimplicit']
        assert properties['dims'] == ['dim1']
        assert properties['units'] == 'm s^-1'
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10])*10.,
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        diagnostics, _ = implicit(state, timedelta(seconds=5))
        assert 'output1_tendency_from_mockimplicit' in diagnostics.keys()
        assert len(
            diagnostics['output1_tendency_from_mockimplicit'].dims) == 1
        assert 'dim1' in diagnostics['output1_tendency_from_mockimplicit'].dims
        assert diagnostics['output1_tendency_from_mockimplicit'].attrs['units'] == 'm s^-1'
        assert np.all(
            diagnostics['output1_tendency_from_mockimplicit'].values == 2.)

    def test_tendencies_in_diagnostics_one_tendency_mismatched_units(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'km'
            }
        }
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]) * 20.,
        }
        with self.assertRaises(InvalidPropertyDictError):
            implicit = MockImplicit(
                input_properties, diagnostic_properties, output_properties,
                diagnostic_output, output_state, tendencies_in_diagnostics=True,
            )

    def test_tendencies_in_diagnostics_one_tendency_mismatched_dims(self):
        input_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim2'],
                'units': 'm'
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]) * 20.,
        }
        with self.assertRaises(InvalidPropertyDictError):
            implicit = MockImplicit(
                input_properties, diagnostic_properties, output_properties,
                diagnostic_output, output_state, tendencies_in_diagnostics=True,
            )

    def test_tendencies_in_diagnostics_one_tendency_with_component_name(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]) * 7.,
        }
        implicit = MockImplicit(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state, tendencies_in_diagnostics=True,
            name='component'
        )
        assert len(implicit.diagnostic_properties) == 1
        assert 'output1_tendency_from_component' in implicit.diagnostic_properties.keys()
        properties = implicit.diagnostic_properties[
            'output1_tendency_from_component']
        assert properties['dims'] == ['dim1']
        assert properties['units'] == 'm s^-1'
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]) * 2.,
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        diagnostics, _ = implicit(state, timedelta(seconds=5))
        assert 'output1_tendency_from_component' in diagnostics.keys()
        assert len(diagnostics['output1_tendency_from_component'].dims) == 1
        assert 'dim1' in diagnostics['output1_tendency_from_component'].dims
        assert diagnostics['output1_tendency_from_component'].attrs['units'] == 'm s^-1'
        assert np.all(diagnostics['output1_tendency_from_component'].values == 1.)


if __name__ == '__main__':
    pytest.main([__file__])
