import pytest
import mock
from sympl import (
    TendencyComponent, Leapfrog, AdamsBashforth, DataArray, SSPRungeKutta, timedelta,
    InvalidPropertyDictError, ImplicitTendencyComponent)
from sympl._core.units import units_are_compatible
import numpy as np
import warnings


def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockEmptyTendencyComponent(TendencyComponent):

    input_properties = {}
    tendency_properties = {}
    diagnostic_properties = {}

    def array_call(self, state):
        return {}, {}


class MockTendencyComponent(TendencyComponent):

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
        super(MockTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self._tendency_output, self._diagnostic_output


class MockImplicitTendencyComponent(ImplicitTendencyComponent):
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
        super(MockImplicitTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        return self._tendency_output, self._diagnostic_output


class PrognosticBase(object):

    prognostic_class = MockTendencyComponent

    def test_given_tendency_not_modified_with_two_components(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        }
        diagnostic_output = {}
        tendency_output_1 = {
            'tend1': np.ones([10]) * 5.
        }
        prognostic1 = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output_1
        )
        tendency_output_2 = {
            'tend1': np.ones([10]) * 5.
        }
        prognostic2 = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output_2
        )
        stepper = self.timestepper_class(
            prognostic1, prognostic2, tendencies_in_diagnostics=True)
        state = {
            'time': timedelta(0),
            'tend1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'},
            )
        }
        _, _ = stepper(state, timedelta(seconds=5))
        assert np.all(tendency_output_1['tend1'] == 5.)
        assert np.all(tendency_output_2['tend1'] == 5.)

    def test_input_state_not_modified_with_two_components(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        }
        diagnostic_output = {}
        tendency_output_1 = {
            'tend1': np.ones([10]) * 5.
        }
        prognostic1 = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output_1
        )
        tendency_output_2 = {
            'tend1': np.ones([10]) * 5.
        }
        prognostic2 = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output_2
        )
        stepper = self.timestepper_class(
            prognostic1, prognostic2, tendencies_in_diagnostics=True)
        state = {
            'time': timedelta(0),
            'tend1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'},
            )
        }
        _, _ = stepper(state, timedelta(seconds=5))
        assert state['tend1'].attrs['units'] == 'm'
        assert np.all(state['tend1'].values == 1.)

    def test_tendencies_in_diagnostics_no_tendency(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {}
        diagnostic_output = {}
        tendency_output = {}
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(
            prognostic, tendencies_in_diagnostics=True)
        state = {'time': timedelta(0)}
        diagnostics, _ = stepper(state, timedelta(seconds=5))
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
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(
            prognostic, tendencies_in_diagnostics=True)
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10])*10.,
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        diagnostics, _ = stepper(state, timedelta(seconds=5))
        tendency_name = 'output1_tendency_from_{}'.format(stepper.__class__.__name__)
        assert tendency_name in diagnostics.keys()
        assert len(
            diagnostics[tendency_name].dims) == 1
        assert 'dim1' in diagnostics[tendency_name].dims
        assert units_are_compatible(diagnostics[tendency_name].attrs['units'], 'm s^-1')
        assert np.allclose(diagnostics[tendency_name].values, 2.)

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
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(
            prognostic, tendencies_in_diagnostics=True, name='component')
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10])*10.,
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        diagnostics, _ = stepper(state, timedelta(seconds=5))
        assert 'output1_tendency_from_component' in diagnostics.keys()
        assert len(
            diagnostics['output1_tendency_from_component'].dims) == 1
        assert 'dim1' in diagnostics['output1_tendency_from_component'].dims
        assert units_are_compatible(diagnostics['output1_tendency_from_component'].attrs['units'], 'm s^-1')
        assert np.allclose(diagnostics['output1_tendency_from_component'].values, 2.)

    def test_copies_untouched_quantities(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            },
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(
            prognostic, tendencies_in_diagnostics=True, name='component')
        untouched_quantity = DataArray(
            np.ones([10])*10.,
            dims=['dim1'],
            attrs={'units': 'J'}
        )
        state = {
            'time': timedelta(0),
            'output1': DataArray(
                np.ones([10])*10.,
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
            'input1': untouched_quantity,
        }
        _, new_state = stepper(state, timedelta(seconds=5))
        assert 'input1' in new_state.keys()
        assert new_state['input1'].dims == untouched_quantity.dims
        assert np.allclose(new_state['input1'].values, 10.)
        assert new_state['input1'].attrs['units'] == 'J'

    def test_stepper_requires_input_for_stepped_quantity(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            },
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(prognostic)
        assert 'output1' in stepper.input_properties.keys()
        assert stepper.input_properties['output1']['dims'] == ['dim1']
        assert units_are_compatible(stepper.input_properties['output1']['units'], 'm')

    def test_stepper_outputs_stepped_quantity(self):
        input_properties = {}
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            },
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(prognostic)
        assert 'output1' in stepper.output_properties.keys()
        assert stepper.output_properties['output1']['dims'] == ['dim1']
        assert units_are_compatible(stepper.output_properties['output1']['units'], 'm')

    def test_stepper_requires_input_for_input_quantity(self):
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 's',
            }
        }
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            },
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(prognostic)
        assert 'input1' in stepper.input_properties.keys()
        assert stepper.input_properties['input1']['dims'] == ['dim1', 'dim2']
        assert units_are_compatible(stepper.input_properties['input1']['units'], 's')
        assert len(stepper.diagnostic_properties) == 0

    def test_stepper_gives_diagnostic_quantity(self):
        input_properties = {}
        diagnostic_properties = {
            'diag1': {
                'dims': ['dim2'],
                'units': '',
            }
        }
        tendency_properties = {
        }
        diagnostic_output = {}
        tendency_output = {
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(
            prognostic, tendencies_in_diagnostics=True, name='component')
        assert 'diag1' in stepper.diagnostic_properties.keys()
        assert stepper.diagnostic_properties['diag1']['dims'] == ['dim2']
        assert units_are_compatible(
            stepper.diagnostic_properties['diag1']['units'], '')
        assert len(stepper.input_properties) == 0
        assert len(stepper.output_properties) == 0

    def test_stepper_gives_diagnostic_tendency_quantity(self):
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 's',
            }
        }
        diagnostic_properties = {}
        tendency_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm/s'
            },
        }
        diagnostic_output = {}
        tendency_output = {
            'output1': np.ones([10]) * 2.,
        }
        prognostic = self.prognostic_class(
            input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output
        )
        stepper = self.timestepper_class(
            prognostic, tendencies_in_diagnostics=True)
        tendency_name = 'output1_tendency_from_{}'.format(stepper.__class__.__name__)
        assert tendency_name in stepper.diagnostic_properties.keys()
        assert len(stepper.diagnostic_properties) == 1
        assert stepper.diagnostic_properties[tendency_name]['dims'] == ['dim1']
        assert units_are_compatible(stepper.input_properties['output1']['units'], 'm')
        assert units_are_compatible(stepper.diagnostic_properties[tendency_name]['units'], 'm/s')


class ImplicitPrognosticBase(PrognosticBase):

    prognostic_class = MockImplicitTendencyComponent


class TimesteppingBase(object):

    timestepper_class = None

    def test_unused_quantities_carried_over(self):
        state = {'time': timedelta(0), 'air_temperature': 273.}
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        timestep = timedelta(seconds=1.)
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}

    def test_timestepper_reveals_prognostics(self):
        prog1 = MockEmptyTendencyComponent()
        prog1.input_properties = {'input1': {'dims': ['dim1'], 'units': 'm'}}
        time_stepper = self.timestepper_class(prog1)
        assert same_list(time_stepper.prognostic_list, (prog1,))

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_float_no_change_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
        assert diagnostics == {}

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_float_no_change_one_step_diagnostic(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': 0.}, {'foo': 'bar'})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
        assert diagnostics == {'foo': 'bar'}

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_float_no_change_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_float_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 1.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 274.}

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_float_one_step_with_units(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'eastward_wind': DataArray(0.02, attrs={'units': 'km/s^2'})}, {})
        state = {'time': timedelta(0), 'eastward_wind': DataArray(1., attrs={'units': 'm/s'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'eastward_wind': DataArray(1., attrs={'units': 'm/s'})}
        assert same_list(new_state.keys(), ['time', 'eastward_wind'])
        assert np.allclose(new_state['eastward_wind'].values, 21.)
        assert new_state['eastward_wind'].attrs['units'] == 'm/s'

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_float_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 1.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 274.}
        state = new_state
        diagnostics, new_state = time_stepper.__call__(new_state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 274.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 275.}
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 275.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 276.}

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_array_no_change_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.zeros((3, 3))}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_array_no_change_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*0.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_array_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*1.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*274.).all()

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_array_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*1.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*274.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(new_state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*274.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*275.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*275.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*276.).all()

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_dataarray_no_change_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                 DataArray(np.zeros((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'].values == np.ones((3, 3))*273.).all()
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'].values == np.ones((3, 3))*273.).all()
        assert len(new_state['air_temperature'].attrs) == 1
        assert new_state['air_temperature'].attrs['units'] == 'K'

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_dataarray_no_change_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                DataArray(np.zeros((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
        state = new_state
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
        state = new_state
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert len(new_state['air_temperature'].attrs) == 1
        assert new_state['air_temperature'].attrs['units'] == 'K'

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_dataarray_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                DataArray(np.ones((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*274.).all()
        assert len(new_state['air_temperature'].attrs) == 1
        assert new_state['air_temperature'].attrs['units'] == 'K'

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_dataarray_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                DataArray(np.ones((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*274.).all()
        state = new_state
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        diagnostics, new_state = time_stepper.__call__(new_state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*274.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*275.).all()
        state = new_state
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*275.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*276.).all()
        assert len(new_state['air_temperature'].attrs) == 1
        assert new_state['air_temperature'].attrs['units'] == 'K'

    @mock.patch.object(MockEmptyTendencyComponent, '__call__')
    def test_array_four_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3)) * 1.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3)) * 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockEmptyTendencyComponent())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3)) * 273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3)) * 274.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(new_state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3)) * 274.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3)) * 275.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3)) * 275.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3)) * 276.).all()
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3)) * 276.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3)) * 277.).all()


class TestAdamsBashforthFirstOrder(TimesteppingBase, PrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 1
        return AdamsBashforth(*args, **kwargs)


class TestAdamsBashforthFirstOrderImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 1
        return AdamsBashforth(*args, **kwargs)


class TestSSPRungeKuttaTwoStep(TimesteppingBase, PrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['stages'] = 2
        return SSPRungeKutta(*args, **kwargs)


class TestSSPRungeKuttaTwoStepImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['stages'] = 2
        return SSPRungeKutta(*args, **kwargs)


class TestSSPRungeKuttaThreeStep(TimesteppingBase, PrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['stages'] = 3
        return SSPRungeKutta(*args, **kwargs)

class TestSSPRungeKuttaThreeStepImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['stages'] = 3
        return SSPRungeKutta(*args, **kwargs)


class TestLeapfrog(TimesteppingBase, PrognosticBase):

    timestepper_class = Leapfrog


class TestLeapfrogImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    timestepper_class = Leapfrog


class TestAdamsBashforthSecondOrder(TimesteppingBase, PrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 2
        return AdamsBashforth(*args, **kwargs)


class TestAdamsBashforthSecondOrderImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 2
        return AdamsBashforth(*args, **kwargs)


class TestAdamsBashforthThirdOrder(TimesteppingBase, PrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 3
        return AdamsBashforth(*args, **kwargs)


class TestAdamsBashforthThirdOrderImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 3
        return AdamsBashforth(*args, **kwargs)


class TestAdamsBashforthFourthOrder(TimesteppingBase, PrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 4
        return AdamsBashforth(*args, **kwargs)


class TestAdamsBashforthFourthOrderImplicitPrognostic(TimesteppingBase, ImplicitPrognosticBase):

    def timestepper_class(self, *args, **kwargs):
        kwargs['order'] = 4
        return AdamsBashforth(*args, **kwargs)


@mock.patch.object(MockEmptyTendencyComponent, '__call__')
def test_leapfrog_float_two_steps_filtered(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
    state = {'time': timedelta(0), 'air_temperature': 273.}
    timestep = timedelta(seconds=1.)
    time_stepper = Leapfrog(MockEmptyTendencyComponent(), asselin_strength=0.5, alpha=1.)
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    assert state == {'time': timedelta(0), 'air_temperature': 273.}
    assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
    state = new_state
    mock_prognostic_call.return_value = ({'air_temperature': 2.}, {})
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    # Asselin filter modifies the current state
    assert state == {'time': timedelta(0), 'air_temperature': 274.}
    assert new_state == {'time': timedelta(0), 'air_temperature': 277.}


@mock.patch.object(MockEmptyTendencyComponent, '__call__')
def test_leapfrog_requires_same_timestep(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
    state = {'time': timedelta(0), 'air_temperature': 273.}
    time_stepper = Leapfrog([MockEmptyTendencyComponent()], asselin_strength=0.5)
    diagnostics, state = time_stepper.__call__(state, timedelta(seconds=1.))
    try:
        time_stepper.__call__(state, timedelta(seconds=2.))
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('Leapfrog must require timestep to be constant')


@mock.patch.object(MockEmptyTendencyComponent, '__call__')
def test_adams_bashforth_requires_same_timestep(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
    state = {'time': timedelta(0), 'air_temperature': 273.}
    time_stepper = AdamsBashforth(MockEmptyTendencyComponent())
    state = time_stepper.__call__(state, timedelta(seconds=1.))
    try:
        time_stepper.__call__(state, timedelta(seconds=2.))
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError(
            'AdamsBashforth must require timestep to be constant')


@mock.patch.object(MockEmptyTendencyComponent, '__call__')
def test_leapfrog_array_two_steps_filtered(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = (
        {'air_temperature': np.ones((3, 3))*0.}, {})
    state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
    timestep = timedelta(seconds=1.)
    time_stepper = Leapfrog(MockEmptyTendencyComponent(), asselin_strength=0.5, alpha=1.)
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    assert same_list(state.keys(), ['time', 'air_temperature'])
    assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
    assert same_list(new_state.keys(), ['time', 'air_temperature'])
    assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
    state = new_state
    mock_prognostic_call.return_value = (
        {'air_temperature': np.ones((3, 3))*2.}, {})
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    # Asselin filter modifies the current state
    assert same_list(state.keys(), ['time', 'air_temperature'])
    assert (state['air_temperature'] == np.ones((3, 3))*274.).all()
    assert same_list(new_state.keys(), ['time', 'air_temperature'])
    assert (new_state['air_temperature'] == np.ones((3, 3))*277.).all()


@mock.patch.object(MockEmptyTendencyComponent, '__call__')
def test_leapfrog_array_two_steps_filtered_williams(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied with a
    Williams factor of alpha=0.5"""
    mock_prognostic_call.return_value = (
        {'air_temperature': np.ones((3, 3))*0.}, {})
    state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
    timestep = timedelta(seconds=1.)
    time_stepper = Leapfrog(MockEmptyTendencyComponent(), asselin_strength=0.5, alpha=0.5)
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    assert same_list(state.keys(), ['time', 'air_temperature'])
    assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
    assert same_list(new_state.keys(), ['time', 'air_temperature'])
    assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()
    state = new_state
    mock_prognostic_call.return_value = (
        {'air_temperature': np.ones((3, 3))*2.}, {})
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    # Asselin filter modifies the current state
    assert same_list(state.keys(), ['time', 'air_temperature'])
    assert (state['air_temperature'] == np.ones((3, 3))*273.5).all()
    assert same_list(new_state.keys(), ['time', 'air_temperature'])
    assert (new_state['air_temperature'] == np.ones((3, 3))*276.5).all()


if __name__ == '__main__':
    pytest.main([__file__])
