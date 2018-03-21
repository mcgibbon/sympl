import pytest
import mock
import numpy as np
import unittest
from sympl import (
    Prognostic, Diagnostic, Monitor, Implicit, ImplicitPrognostic,
    datetime, timedelta, DataArray, InvalidPropertyDictError
)

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
        self._diagnostic_output = diagnostic_output
        self._tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        super(MockPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self._tendency_output, self._diagnostic_output


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
        self._diagnostic_output = diagnostic_output
        self._tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicitPrognostic, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self._tendency_output, self._diagnostic_output


class MockDiagnostic(Diagnostic):

    input_properties = None
    diagnostic_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, diagnostic_output,
            **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self._diagnostic_output = diagnostic_output
        self.times_called = 0
        self.state_given = None
        super(MockDiagnostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self._diagnostic_output


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
        self._diagnostic_output = diagnostic_output
        self._state_output = state_output
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicit, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self._diagnostic_output, self._state_output


class MockMonitor(Monitor):

    def store(self, state):
        return


class PrognosticTests(unittest.TestCase):

    component_class = MockPrognostic

    def call_component(self, component, state):
        return component(state)

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

    def test_input_no_transformations(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(prognostic, state)
        assert len(prognostic.state_given) == 2
        assert 'time' in prognostic.state_given.keys()
        assert 'input1' in prognostic.state_given.keys()
        assert isinstance(prognostic.state_given['input1'], np.ndarray)
        assert np.all(prognostic.state_given['input1'] == np.ones([10]))

    def test_input_converts_units(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'km'}
            )
        }
        _, _ = self.call_component(prognostic, state)
        assert len(prognostic.state_given) == 2
        assert 'time' in prognostic.state_given.keys()
        assert 'input1' in prognostic.state_given.keys()
        assert isinstance(prognostic.state_given['input1'], np.ndarray)
        assert np.all(prognostic.state_given['input1'] == np.ones([10])*1000.)

    def test_input_collects_one_dimension(self):
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(prognostic, state)
        assert len(prognostic.state_given) == 2
        assert 'time' in prognostic.state_given.keys()
        assert 'input1' in prognostic.state_given.keys()
        assert isinstance(prognostic.state_given['input1'], np.ndarray)
        assert np.all(prognostic.state_given['input1'] == np.ones([10]))

    def test_input_is_aliased(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'in1',
            }
        }
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(prognostic, state)
        assert len(prognostic.state_given) == 2
        assert 'time' in prognostic.state_given.keys()
        assert 'in1' in prognostic.state_given.keys()
        assert isinstance(prognostic.state_given['in1'], np.ndarray)
        assert np.all(prognostic.state_given['in1'] == np.ones([10]))

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

    def test_diagnostics_no_transformations(self):
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        tendency_properties = {}
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        _, diagnostics = self.call_component(prognostic, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_with_alias(self):
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'out1',
            }
        }
        tendency_properties = {}
        diagnostic_output = {
            'out1': np.ones([10]),
        }
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        state = {'time': timedelta(0)}
        _, diagnostics = self.call_component(prognostic, state)
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
        tendency_properties = {}
        diagnostic_output = {
            'out1': np.ones([10]),
        }
        tendency_output = {}
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
        _, diagnostics = self.call_component(prognostic, state)
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
        tendency_properties = {}
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        tendency_output = {}
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
        _, diagnostics = self.call_component(prognostic, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_input_scaling(self):
        input_scale_factors = {'input1': 2.}
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output,
            input_scale_factors=input_scale_factors
        )
        assert prognostic.tendency_scale_factors == {}
        assert prognostic.diagnostic_scale_factors == {}
        assert len(prognostic.input_scale_factors) == 1
        assert 'input1' in prognostic.input_scale_factors.keys()
        assert prognostic.input_scale_factors['input1'] == 2.
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(prognostic, state)
        assert len(prognostic.state_given) == 2
        assert 'time' in prognostic.state_given.keys()
        assert 'input1' in prognostic.state_given.keys()
        assert isinstance(prognostic.state_given['input1'], np.ndarray)
        assert np.all(prognostic.state_given['input1'] == np.ones([10])*2.)

    def test_tendency_scaling(self):
        tendency_scale_factors = {'output1': 3.}
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
            'output1': np.ones([10]),
        }
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output,
            tendency_scale_factors=tendency_scale_factors,
        )
        assert prognostic.input_scale_factors == {}
        assert prognostic.diagnostic_scale_factors == {}
        assert len(prognostic.tendency_scale_factors) == 1
        assert 'output1' in prognostic.tendency_scale_factors.keys()
        assert prognostic.tendency_scale_factors['output1'] == 3.
        state = {'time': timedelta(0)}
        tendencies, _ = self.call_component(prognostic, state)
        assert len(tendencies) == 1
        assert 'output1' in tendencies.keys()
        assert isinstance(tendencies['output1'], DataArray)
        assert len(tendencies['output1'].dims) == 1
        assert 'dim1' in tendencies['output1'].dims
        assert 'units' in tendencies['output1'].attrs
        assert tendencies['output1'].attrs['units'] == 'm/s'
        assert np.all(tendencies['output1'].values == np.ones([10])*3.)

    def test_diagnostics_scaling(self):
        diagnostic_scale_factors = {'output1': 0.}
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        tendency_properties = {}
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output,
            diagnostic_scale_factors=diagnostic_scale_factors,
        )
        assert prognostic.tendency_scale_factors == {}
        assert prognostic.input_scale_factors == {}
        assert len(prognostic.diagnostic_scale_factors) == 1
        assert 'output1' in prognostic.diagnostic_scale_factors.keys()
        assert prognostic.diagnostic_scale_factors['output1'] == 0.
        state = {'time': timedelta(0)}
        _, diagnostics = self.call_component(prognostic, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10])*0.)

    def test_update_interval_on_timedelta(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, update_interval=timedelta(seconds=10)
        )
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=0)})
        assert prognostic.times_called == 1
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=0)})
        assert prognostic.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=5)})
        assert prognostic.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=10)})
        assert prognostic.times_called == 2, 'should re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=15)})
        assert prognostic.times_called == 2, 'should not re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=20)})
        assert prognostic.times_called == 3, 'should re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=30)})
        assert prognostic.times_called == 4, 'should re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=45)})
        assert prognostic.times_called == 5, 'should re-compute output'
        _, _ = self.call_component(prognostic, {'time': timedelta(seconds=50)})
        assert prognostic.times_called == 5, 'should not re-compute output'

    def test_update_interval_on_datetime(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output,
            update_interval=timedelta(seconds=10)
        )
        dt = datetime(2010, 1, 1)
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=0)})
        assert prognostic.times_called == 1
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=0)})
        assert prognostic.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=5)})
        assert prognostic.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=10)})
        assert prognostic.times_called == 2, 'should re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=15)})
        assert prognostic.times_called == 2, 'should not re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=20)})
        assert prognostic.times_called == 3, 'should re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=30)})
        assert prognostic.times_called == 4, 'should re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=45)})
        assert prognostic.times_called == 5, 'should re-compute output'
        _, _ = self.call_component(
            prognostic, {'time': dt + timedelta(seconds=50)})
        assert prognostic.times_called == 5, 'should not re-compute output'


class ImplicitPrognosticTests(PrognosticTests):

    component_class = MockImplicitPrognostic

    def call_component(self, component, state):
        return component(state, timedelta(seconds=1))

    def test_timedelta_is_passed(self):
        prognostic = MockImplicitPrognostic({}, {}, {}, {}, {})
        tendencies, diagnostics = prognostic(
            {'time': timedelta(seconds=0)}, timedelta(seconds=5))
        assert tendencies == {}
        assert diagnostics == {}
        assert prognostic.timestep_given == timedelta(seconds=5)
        assert prognostic.times_called == 1


class DiagnosticTests(unittest.TestCase):

    component_class = MockDiagnostic

    def call_component(self, component, state):
        return component(state)

    def test_empty_diagnostic(self):
        diagnostic = self.component_class({}, {}, {})
        diagnostics = diagnostic({'time': timedelta(seconds=0)})
        assert diagnostics == {}
        assert len(diagnostic.state_given) == 1
        assert 'time' in diagnostic.state_given.keys()
        assert diagnostic.state_given['time'] == timedelta(seconds=0)
        assert diagnostic.times_called == 1

    def test_input_no_transformations(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties, diagnostic_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _ = diagnostic(state)
        assert len(diagnostic.state_given) == 2
        assert 'time' in diagnostic.state_given.keys()
        assert 'input1' in diagnostic.state_given.keys()
        assert isinstance(diagnostic.state_given['input1'], np.ndarray)
        assert np.all(diagnostic.state_given['input1'] == np.ones([10]))

    def test_input_converts_units(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'km'}
            )
        }
        _ = diagnostic(state)
        assert len(diagnostic.state_given) == 2
        assert 'time' in diagnostic.state_given.keys()
        assert 'input1' in diagnostic.state_given.keys()
        assert isinstance(diagnostic.state_given['input1'], np.ndarray)
        assert np.all(diagnostic.state_given['input1'] == np.ones([10])*1000.)

    def test_input_collects_one_dimension(self):
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _ = diagnostic(state)
        assert len(diagnostic.state_given) == 2
        assert 'time' in diagnostic.state_given.keys()
        assert 'input1' in diagnostic.state_given.keys()
        assert isinstance(diagnostic.state_given['input1'], np.ndarray)
        assert np.all(diagnostic.state_given['input1'] == np.ones([10]))

    def test_input_is_aliased(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'in1',
            }
        }
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _ = diagnostic(state)
        assert len(diagnostic.state_given) == 2
        assert 'time' in diagnostic.state_given.keys()
        assert 'in1' in diagnostic.state_given.keys()
        assert isinstance(diagnostic.state_given['in1'], np.ndarray)
        assert np.all(diagnostic.state_given['in1'] == np.ones([10]))

    def test_diagnostics_no_transformations(self):
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {'time': timedelta(0)}
        diagnostics = diagnostic(state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_with_alias(self):
        input_properties = {}
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
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {'time': timedelta(0)}
        diagnostics = diagnostic(state)
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
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        diagnostics = diagnostic(state)
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
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties,
            diagnostic_output
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        diagnostics = diagnostic(state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_input_scaling(self):
        input_scale_factors = {'input1': 2.}
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = self.component_class(
            input_properties, diagnostic_properties,
            diagnostic_output,
            input_scale_factors=input_scale_factors
        )
        assert diagnostic.diagnostic_scale_factors == {}
        assert len(diagnostic.input_scale_factors) == 1
        assert 'input1' in diagnostic.input_scale_factors.keys()
        assert diagnostic.input_scale_factors['input1'] == 2.
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _ = self.call_component(diagnostic, state)
        assert len(diagnostic.state_given) == 2
        assert 'time' in diagnostic.state_given.keys()
        assert 'input1' in diagnostic.state_given.keys()
        assert isinstance(diagnostic.state_given['input1'], np.ndarray)
        assert np.all(diagnostic.state_given['input1'] == np.ones([10]) * 2.)

    def test_diagnostics_scaling(self):
        diagnostic_scale_factors = {'output1': 0.}
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        diagnostic = self.component_class(
            input_properties, diagnostic_properties,
            diagnostic_output,
            diagnostic_scale_factors=diagnostic_scale_factors,
        )
        assert diagnostic.input_scale_factors == {}
        assert len(diagnostic.diagnostic_scale_factors) == 1
        assert 'output1' in diagnostic.diagnostic_scale_factors.keys()
        assert diagnostic.diagnostic_scale_factors['output1'] == 0.
        state = {'time': timedelta(0)}
        diagnostics = self.call_component(diagnostic, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]) * 0.)

    def test_update_interval_on_timedelta(self):
        input_properties = {}
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = self.component_class(
            input_properties, diagnostic_properties,
            diagnostic_output,
            update_interval=timedelta(seconds=10)
        )
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=0)})
        assert diagnostic.times_called == 1
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=0)})
        assert diagnostic.times_called == 1, 'should not re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=5)})
        assert diagnostic.times_called == 1, 'should not re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=10)})
        assert diagnostic.times_called == 2, 'should re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=15)})
        assert diagnostic.times_called == 2, 'should not re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=20)})
        assert diagnostic.times_called == 3, 'should re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=30)})
        assert diagnostic.times_called == 4, 'should re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=45)})
        assert diagnostic.times_called == 5, 'should re-compute output'
        _ = self.call_component(diagnostic, {'time': timedelta(seconds=50)})
        assert diagnostic.times_called == 5, 'should not re-compute output'

    def test_update_interval_on_datetime(self):
        input_properties = {}
        diagnostic_properties = {}
        diagnostic_output = {}
        diagnostic = self.component_class(
            input_properties, diagnostic_properties,
            diagnostic_output,
            update_interval=timedelta(seconds=10)
        )
        dt = datetime(2010, 1, 1)
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=0)})
        assert diagnostic.times_called == 1
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=0)})
        assert diagnostic.times_called == 1, 'should not re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=5)})
        assert diagnostic.times_called == 1, 'should not re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=10)})
        assert diagnostic.times_called == 2, 'should re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=15)})
        assert diagnostic.times_called == 2, 'should not re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=20)})
        assert diagnostic.times_called == 3, 'should re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=30)})
        assert diagnostic.times_called == 4, 'should re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=45)})
        assert diagnostic.times_called == 5, 'should re-compute output'
        _ = self.call_component(
            diagnostic, {'time': dt + timedelta(seconds=50)})
        assert diagnostic.times_called == 5, 'should not re-compute output'


class ImplicitTests(unittest.TestCase):

    component_class = MockImplicit

    def call_component(self, component, state):
        return component(state, timedelta(seconds=1))

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

    def test_timedelta_is_passed(self):
        implicit = MockImplicit({}, {}, {}, {}, {})
        tendencies, diagnostics = implicit(
            {'time': timedelta(seconds=0)}, timedelta(seconds=5))
        assert tendencies == {}
        assert diagnostics == {}
        assert implicit.timestep_given == timedelta(seconds=5)
        assert implicit.times_called == 1

    def test_input_no_transformations(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(implicit, state)
        assert len(implicit.state_given) == 2
        assert 'time' in implicit.state_given.keys()
        assert 'input1' in implicit.state_given.keys()
        assert isinstance(implicit.state_given['input1'], np.ndarray)
        assert np.all(implicit.state_given['input1'] == np.ones([10]))

    def test_input_converts_units(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'km'}
            )
        }
        _, _ = self.call_component(implicit, state)
        assert len(implicit.state_given) == 2
        assert 'time' in implicit.state_given.keys()
        assert 'input1' in implicit.state_given.keys()
        assert isinstance(implicit.state_given['input1'], np.ndarray)
        assert np.all(implicit.state_given['input1'] == np.ones([10])*1000.)

    def test_input_collects_one_dimension(self):
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(implicit, state)
        assert len(implicit.state_given) == 2
        assert 'time' in implicit.state_given.keys()
        assert 'input1' in implicit.state_given.keys()
        assert isinstance(implicit.state_given['input1'], np.ndarray)
        assert np.all(implicit.state_given['input1'] == np.ones([10]))

    def test_input_is_aliased(self):
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'in1',
            }
        }
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        prognostic = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(prognostic, state)
        assert len(prognostic.state_given) == 2
        assert 'time' in prognostic.state_given.keys()
        assert 'in1' in prognostic.state_given.keys()
        assert isinstance(prognostic.state_given['in1'], np.ndarray)
        assert np.all(prognostic.state_given['in1'] == np.ones([10]))

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

    def test_diagnostics_no_transformations(self):
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        output_properties = {}
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {'time': timedelta(0)}
        diagnostics, _ = self.call_component(implicit, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_diagnostics_with_alias(self):
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
                'alias': 'out1',
            }
        }
        output_properties = {}
        diagnostic_output = {
            'out1': np.ones([10]),
        }
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state
        )
        state = {'time': timedelta(0)}
        diagnostics, _ = self.call_component(implicit, state)
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
        output_properties = {}
        diagnostic_output = {
            'out1': np.ones([10]),
        }
        output_state = {}
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
        diagnostics, _ = self.call_component(implicit, state)
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
        output_properties = {}
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        output_state = {}
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
        diagnostics, _ = self.call_component(implicit, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]))

    def test_input_scaling(self):
        input_scale_factors = {'input1': 2.}
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state,
            input_scale_factors=input_scale_factors
        )
        assert implicit.output_scale_factors == {}
        assert implicit.diagnostic_scale_factors == {}
        assert len(implicit.input_scale_factors) == 1
        assert 'input1' in implicit.input_scale_factors.keys()
        assert implicit.input_scale_factors['input1'] == 2.
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        _, _ = self.call_component(implicit, state)
        assert len(implicit.state_given) == 2
        assert 'time' in implicit.state_given.keys()
        assert 'input1' in implicit.state_given.keys()
        assert isinstance(implicit.state_given['input1'], np.ndarray)
        assert np.all(implicit.state_given['input1'] == np.ones([10]) * 2.)

    def test_output_scaling(self):
        output_scale_factors = {'output1': 3.}
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            }
        }
        diagnostic_output = {}
        output_state = {
            'output1': np.ones([10]),
        }
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state,
            output_scale_factors=output_scale_factors,
        )
        assert implicit.input_scale_factors == {}
        assert implicit.diagnostic_scale_factors == {}
        assert len(implicit.output_scale_factors) == 1
        assert 'output1' in implicit.output_scale_factors.keys()
        assert implicit.output_scale_factors['output1'] == 3.
        state = {'time': timedelta(0)}
        _, output = self.call_component(implicit, state)
        assert len(output) == 1
        assert 'output1' in output.keys()
        assert isinstance(output['output1'], DataArray)
        assert len(output['output1'].dims) == 1
        assert 'dim1' in output['output1'].dims
        assert 'units' in output['output1'].attrs
        assert output['output1'].attrs['units'] == 'm/s'
        assert np.all(output['output1'].values == np.ones([10]) * 3.)

    def test_diagnostics_scaling(self):
        diagnostic_scale_factors = {'output1': 0.}
        input_properties = {}
        diagnostic_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        output_properties = {}
        diagnostic_output = {
            'output1': np.ones([10]),
        }
        tendency_output = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, tendency_output,
            diagnostic_scale_factors=diagnostic_scale_factors,
        )
        assert implicit.output_scale_factors == {}
        assert implicit.input_scale_factors == {}
        assert len(implicit.diagnostic_scale_factors) == 1
        assert 'output1' in implicit.diagnostic_scale_factors.keys()
        assert implicit.diagnostic_scale_factors['output1'] == 0.
        state = {'time': timedelta(0)}
        diagnostics, _ = self.call_component(implicit, state)
        assert len(diagnostics) == 1
        assert 'output1' in diagnostics.keys()
        assert isinstance(diagnostics['output1'], DataArray)
        assert len(diagnostics['output1'].dims) == 1
        assert 'dim1' in diagnostics['output1'].dims
        assert 'units' in diagnostics['output1'].attrs
        assert diagnostics['output1'].attrs['units'] == 'm'
        assert np.all(diagnostics['output1'].values == np.ones([10]) * 0.)

    def test_update_interval_on_timedelta(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state,
            update_interval=timedelta(seconds=10)
        )
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=0)})
        assert implicit.times_called == 1
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=0)})
        assert implicit.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=5)})
        assert implicit.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=10)})
        assert implicit.times_called == 2, 'should re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=15)})
        assert implicit.times_called == 2, 'should not re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=20)})
        assert implicit.times_called == 3, 'should re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=30)})
        assert implicit.times_called == 4, 'should re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=45)})
        assert implicit.times_called == 5, 'should re-compute output'
        _, _ = self.call_component(implicit, {'time': timedelta(seconds=50)})
        assert implicit.times_called == 5, 'should not re-compute output'

    def test_update_interval_on_datetime(self):
        input_properties = {}
        diagnostic_properties = {}
        output_properties = {}
        diagnostic_output = {}
        output_state = {}
        implicit = self.component_class(
            input_properties, diagnostic_properties, output_properties,
            diagnostic_output, output_state,
            update_interval=timedelta(seconds=10)
        )
        dt = datetime(2010, 1, 1)
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=0)})
        assert implicit.times_called == 1
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=0)})
        assert implicit.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=5)})
        assert implicit.times_called == 1, 'should not re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=10)})
        assert implicit.times_called == 2, 'should re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=15)})
        assert implicit.times_called == 2, 'should not re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=20)})
        assert implicit.times_called == 3, 'should re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=30)})
        assert implicit.times_called == 4, 'should re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=45)})
        assert implicit.times_called == 5, 'should re-compute output'
        _, _ = self.call_component(
            implicit, {'time': dt + timedelta(seconds=50)})
        assert implicit.times_called == 5, 'should not re-compute output'

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
