from datetime import timedelta, datetime
import unittest
from sympl import (
    Prognostic, Implicit, Diagnostic, UpdateFrequencyWrapper, ScalingWrapper,
    TendencyInDiagnosticsWrapper, TimeDifferencingWrapper, DataArray
)
import pytest
from numpy.testing import assert_allclose
from copy import deepcopy


class MockPrognostic(Prognostic):

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state):
        self._num_updates += 1
        return {}, {'num_updates': self._num_updates}


class MockImplicit(Implicit):

    output_properties = {
        'value': {
            'dims': [],
            'units': 'm'
        }
    }

    diagnostic_properties = {
        'num_updates': {
            'dims': [],
            'units': ''
        }
    }

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state, timestep):
        self._num_updates += 1

        return (
            {'num_updates': DataArray([self._num_updates], attrs={'units': ''})},
            {'value': DataArray([1], attrs={'units': 'm'})})


class MockImplicitThatExpects(Implicit):

    input_properties = {'expected_field': {}}
    output_properties = {'expected_field': {}}
    diagnostic_properties = {'expected_field': {}}

    def __init__(self, expected_value):
        self._expected_value = expected_value

    def __call__(self, state, timestep):

        input_value = state['expected_field']
        if input_value != self._expected_value:
            raise ValueError(
                'Expected {}, but got {}'.format(self._expected_value, input_value))

        return deepcopy(state), state


class MockPrognosticThatExpects(Prognostic):

    input_properties = {'expected_field': {}}
    tendency_properties = {'expected_field': {}}
    diagnostic_properties = {'expected_field': {}}

    def __init__(self, expected_value):
        self._expected_value = expected_value

    def __call__(self, state):

        input_value = state['expected_field']
        if input_value != self._expected_value:
            raise ValueError(
                'Expected {}, but got {}'.format(self._expected_value, input_value))

        return deepcopy(state), state


class MockDiagnosticThatExpects(Diagnostic):

    input_properties = {'expected_field': {}}
    diagnostic_properties = {'expected_field': {}}

    def __init__(self, expected_value):
        self._expected_value = expected_value

    def __call__(self, state):

        input_value = state['expected_field']
        if input_value != self._expected_value:
            raise ValueError(
                'Expected {}, but got {}'.format(self._expected_value, input_value))

        return state


class TimeDifferencingTests(unittest.TestCase):

    def setUp(self):
        self.implicit = MockImplicit()
        self.wrapped = TimeDifferencingWrapper(self.implicit)
        self.state = {
            'value': DataArray([0], attrs={'units': 'm'})
        }

    def tearDown(self):
        self.component = None

    def testWrapperCallsImplicit(self):
        tendencies, diagnostics = self.wrapped(self.state, timedelta(seconds=1))
        assert diagnostics['num_updates'].values[0] == 1
        tendencies, diagnostics = self.wrapped(self.state, timedelta(seconds=1))
        assert diagnostics['num_updates'].values[0] == 2
        assert len(diagnostics.keys()) == 1

    def testWrapperComputesTendency(self):
        tendencies, diagnostics = self.wrapped(self.state, timedelta(seconds=1))
        assert len(tendencies.keys()) == 1
        assert 'value' in tendencies.keys()
        assert isinstance(tendencies['value'], DataArray)
        assert_allclose(tendencies['value'].to_units('m s^-1').values[0], 1.)
        assert_allclose(tendencies['value'].values[0], 1.)

    def testWrapperComputesTendencyWithUnitConversion(self):
        state = {
            'value': DataArray([0.011], attrs={'units': 'km'})
        }
        tendencies, diagnostics = self.wrapped(state, timedelta(seconds=5))
        assert 'value' in tendencies.keys()
        assert isinstance(tendencies['value'], DataArray)
        assert_allclose(tendencies['value'].to_units('m s^-1').values[0], -2)
        assert_allclose(tendencies['value'].values[0], -2.)
        assert_allclose(tendencies['value'].to_units('km s^-1').values[0], -0.002)


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


def test_scaled_component_wrong_type():
    class WrongType(object):
        def __init__(self):
            self.a = 1

    wrong_component = WrongType()

    with pytest.raises(TypeError) as excinfo:
        component = ScalingWrapper(wrong_component)

    assert 'either of type Implicit' in str(excinfo.value)


def test_scaled_implicit_inputs():
    implicit = ScalingWrapper(
        MockImplicitThatExpects(2.0),
        input_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    diagnostics, new_state = implicit(state)

    assert new_state['expected_field'] == 2.0
    assert diagnostics['expected_field'] == 2.0


def test_scaled_implicit_outputs():
    implicit = ScalingWrapper(
        MockImplicitThatExpects(4.0),
        output_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    diagnostics, new_state = implicit(state)

    assert new_state['expected_field'] == 2.0
    assert diagnostics['expected_field'] == 4.0


def test_scaled_implicit_diagnostics():
    implicit = ScalingWrapper(
        MockImplicitThatExpects(4.0),
        diagnostic_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    diagnostics, new_state = implicit(state)

    assert diagnostics['expected_field'] == 2.0
    assert new_state['expected_field'] == 4.0


def test_scaled_implicit_created_with_wrong_input_field():
    with pytest.raises(ValueError) as excinfo:
        implicit = ScalingWrapper(
            MockImplicitThatExpects(2.0),
            input_scale_factors={'abcd': 0.5})

    assert 'not a valid input' in str(excinfo.value)


def test_scaled_implicit_created_with_wrong_output_field():
    with pytest.raises(ValueError) as excinfo:
        implicit = ScalingWrapper(
            MockImplicitThatExpects(2.0),
            output_scale_factors={'abcd': 0.5})

    assert 'not a valid output' in str(excinfo.value)


def test_scaled_implicit_created_with_wrong_diagnostic_field():
    with pytest.raises(ValueError) as excinfo:
        implicit = ScalingWrapper(
            MockImplicitThatExpects(2.0),
            diagnostic_scale_factors={'abcd': 0.5})

    assert 'not a valid diagnostic' in str(excinfo.value)


def test_scaled_prognostic_inputs():
    prognostic = ScalingWrapper(
        MockPrognosticThatExpects(2.0),
        input_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    tendencies, diagnostics = prognostic(state)

    assert tendencies['expected_field'] == 2.0
    assert diagnostics['expected_field'] == 2.0


def test_scaled_prognostic_tendencies():
    prognostic = ScalingWrapper(
        MockPrognosticThatExpects(4.0),
        tendency_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    tendencies, diagnostics = prognostic(state)

    assert tendencies['expected_field'] == 2.0
    assert diagnostics['expected_field'] == 4.0


def test_scaled_prognostic_diagnostics():
    prognostic = ScalingWrapper(
        MockPrognosticThatExpects(4.0),
        diagnostic_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    tendencies, diagnostics = prognostic(state)

    assert tendencies['expected_field'] == 4.0
    assert diagnostics['expected_field'] == 2.0


def test_scaled_prognostic_with_wrong_tendency_field():
    with pytest.raises(ValueError) as excinfo:
        prognostic = ScalingWrapper(
            MockPrognosticThatExpects(4.0),
            tendency_scale_factors={'abcd': 0.5})

    assert 'not a valid tendency' in str(excinfo.value)


def test_scaled_diagnostic_inputs():
    diagnostic = ScalingWrapper(
        MockDiagnosticThatExpects(2.0),
        input_scale_factors={'expected_field': 0.5})

    state = {'expected_field': 4.0}

    diagnostics = diagnostic(state)

    assert diagnostics['expected_field'] == 2.0



def test_scaled_diagnostic_diagnostics():

    diagnostic = ScalingWrapper(
        MockDiagnosticThatExpects(4.0),
        diagnostic_scale_factors = {'expected_field': 0.5})

    state = {'expected_field': 4.0}

    diagnostics = diagnostic(state)

    assert diagnostics['expected_field'] == 2.0


def test_scaled_component_type_wrongly_modified():

    diagnostic = ScalingWrapper(
        MockDiagnosticThatExpects(4.0),
        diagnostic_scale_factors = {'expected_field': 0.5})

    state = {'expected_field': 4.0}

    diagnostic._component_type = 'abcd'

    with pytest.raises(ValueError) as excinfo:
        diagnostics = diagnostic(state)

    assert 'bug in ScalingWrapper' in str(excinfo.value)

if __name__ == '__main__':
    pytest.main([__file__])
