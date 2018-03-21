import pytest
import mock
from sympl import (
    Prognostic, Leapfrog, AdamsBashforth, DataArray, SSPRungeKutta, timedelta)
import numpy as np


def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockPrognostic(Prognostic):

    input_properties = {}
    tendency_properties = {}
    diagnostic_properties = {}

    def array_call(self, state):
        return {}, {}


class TimesteppingBase(object):

    timestepper_class = None

    def test_unused_quantities_carried_over(self):
        state = {'time': timedelta(0), 'air_temperature': 273.}
        time_stepper = self.timestepper_class(MockPrognostic())
        timestep = timedelta(seconds=1.)
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}

    def test_timestepper_reveals_prognostics(self):
        prog1 = MockPrognostic()
        prog1.input_properties = {'input1': {}}
        time_stepper = self.timestepper_class(prog1)
        assert same_list(time_stepper.prognostic_list, (prog1,))

    @mock.patch.object(MockPrognostic, '__call__')
    def test_float_no_change_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
        assert diagnostics == {}

    @mock.patch.object(MockPrognostic, '__call__')
    def test_float_no_change_one_step_diagnostic(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': 0.}, {'foo': 'bar'})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
        assert diagnostics == {'foo': 'bar'}

    @mock.patch.object(MockPrognostic, '__call__')
    def test_float_no_change_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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

    @mock.patch.object(MockPrognostic, '__call__')
    def test_float_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 1.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'air_temperature': 273.}
        assert new_state == {'time': timedelta(0), 'air_temperature': 274.}

    @mock.patch.object(MockPrognostic, '__call__')
    def test_float_one_step_with_units(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'eastward_wind': DataArray(0.02, attrs={'units': 'km/s^2'})}, {})
        state = {'time': timedelta(0), 'eastward_wind': DataArray(1., attrs={'units': 'm/s'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert state == {'time': timedelta(0), 'eastward_wind': DataArray(1., attrs={'units': 'm/s'})}
        assert same_list(new_state.keys(), ['time', 'eastward_wind'])
        assert np.allclose(new_state['eastward_wind'].values, 21.)
        assert new_state['eastward_wind'].attrs['units'] == 'm/s'

    @mock.patch.object(MockPrognostic, '__call__')
    def test_float_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = ({'air_temperature': 1.}, {})
        state = {'time': timedelta(0), 'air_temperature': 273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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

    @mock.patch.object(MockPrognostic, '__call__')
    def test_array_no_change_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.zeros((3, 3))}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*273.).all()

    @mock.patch.object(MockPrognostic, '__call__')
    def test_array_no_change_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*0.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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

    @mock.patch.object(MockPrognostic, '__call__')
    def test_array_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*1.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*274.).all()

    @mock.patch.object(MockPrognostic, '__call__')
    def test_array_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*1.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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

    @mock.patch.object(MockPrognostic, '__call__')
    def test_dataarray_no_change_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                 DataArray(np.zeros((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'].values == np.ones((3, 3))*273.).all()
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'].values == np.ones((3, 3))*273.).all()
        assert len(new_state['air_temperature'].attrs) == 1
        assert new_state['air_temperature'].attrs['units'] == 'K'

    @mock.patch.object(MockPrognostic, '__call__')
    def test_dataarray_no_change_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                DataArray(np.zeros((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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

    @mock.patch.object(MockPrognostic, '__call__')
    def test_dataarray_one_step(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                DataArray(np.ones((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*273.).all()
        assert len(state['air_temperature'].attrs) == 1
        assert state['air_temperature'].attrs['units'] == 'K'
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*274.).all()
        assert len(new_state['air_temperature'].attrs) == 1
        assert new_state['air_temperature'].attrs['units'] == 'K'

    @mock.patch.object(MockPrognostic, '__call__')
    def test_dataarray_three_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature':
                DataArray(np.ones((3, 3)), attrs={'units': 'K/s'})},
            {})
        state = {'time': timedelta(0), 'air_temperature': DataArray(np.ones((3, 3))*273.,
                                              attrs={'units': 'K'})}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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


class TestSSPRungeKuttaTwoStep(TimesteppingBase):

    def timestepper_class(self, *args):
        return SSPRungeKutta(*args, stages=2)


class TestSSPRungeKuttaThreeStep(TimesteppingBase):
    def timestepper_class(self, *args):
        return SSPRungeKutta(*args, stages=3)


class TestLeapfrog(TimesteppingBase):
    timestepper_class = Leapfrog


class TestAdamsBashforthSecondOrder(TimesteppingBase):
    def timestepper_class(self, *args):
        return AdamsBashforth(*args, order=2)


class TestAdamsBashforthThirdOrder(TimesteppingBase):
    def timestepper_class(self, *args):
        return AdamsBashforth(*args, order=3)


class TestAdamsBashforthFourthOrder(TimesteppingBase):
    def timestepper_class(self, *args):
        return AdamsBashforth(*args, order=4)

    @mock.patch.object(MockPrognostic, '__call__')
    def test_array_four_steps(self, mock_prognostic_call):
        mock_prognostic_call.return_value = (
            {'air_temperature': np.ones((3, 3))*1.}, {})
        state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
        timestep = timedelta(seconds=1.)
        time_stepper = self.timestepper_class(MockPrognostic())
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
        state = new_state
        diagnostics, new_state = time_stepper.__call__(state, timestep)
        assert same_list(state.keys(), ['time', 'air_temperature'])
        assert (state['air_temperature'] == np.ones((3, 3))*276.).all()
        assert same_list(new_state.keys(), ['time', 'air_temperature'])
        assert (new_state['air_temperature'] == np.ones((3, 3))*277.).all()


@mock.patch.object(MockPrognostic, '__call__')
def test_leapfrog_float_two_steps_filtered(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
    state = {'time': timedelta(0), 'air_temperature': 273.}
    timestep = timedelta(seconds=1.)
    time_stepper = Leapfrog(MockPrognostic(), asselin_strength=0.5, alpha=1.)
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    assert state == {'time': timedelta(0), 'air_temperature': 273.}
    assert new_state == {'time': timedelta(0), 'air_temperature': 273.}
    state = new_state
    mock_prognostic_call.return_value = ({'air_temperature': 2.}, {})
    diagnostics, new_state = time_stepper.__call__(state, timestep)
    # Asselin filter modifies the current state
    assert state == {'time': timedelta(0), 'air_temperature': 274.}
    assert new_state == {'time': timedelta(0), 'air_temperature': 277.}


@mock.patch.object(MockPrognostic, '__call__')
def test_leapfrog_requires_same_timestep(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
    state = {'time': timedelta(0), 'air_temperature': 273.}
    time_stepper = Leapfrog([MockPrognostic()], asselin_strength=0.5)
    diagnostics, state = time_stepper.__call__(state, timedelta(seconds=1.))
    try:
        time_stepper.__call__(state, timedelta(seconds=2.))
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('Leapfrog must require timestep to be constant')


@mock.patch.object(MockPrognostic, '__call__')
def test_adams_bashforth_requires_same_timestep(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = ({'air_temperature': 0.}, {})
    state = {'time': timedelta(0), 'air_temperature': 273.}
    time_stepper = AdamsBashforth(MockPrognostic())
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


@mock.patch.object(MockPrognostic, '__call__')
def test_leapfrog_array_two_steps_filtered(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied"""
    mock_prognostic_call.return_value = (
        {'air_temperature': np.ones((3, 3))*0.}, {})
    state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
    timestep = timedelta(seconds=1.)
    time_stepper = Leapfrog(MockPrognostic(), asselin_strength=0.5, alpha=1.)
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


@mock.patch.object(MockPrognostic, '__call__')
def test_leapfrog_array_two_steps_filtered_williams(mock_prognostic_call):
    """Test that the Asselin filter is being correctly applied with a
    Williams factor of alpha=0.5"""
    mock_prognostic_call.return_value = (
        {'air_temperature': np.ones((3, 3))*0.}, {})
    state = {'time': timedelta(0), 'air_temperature': np.ones((3, 3))*273.}
    timestep = timedelta(seconds=1.)
    time_stepper = Leapfrog(MockPrognostic(), asselin_strength=0.5, alpha=0.5)
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
