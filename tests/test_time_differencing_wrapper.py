from datetime import timedelta, datetime
import unittest
from sympl import (
    TendencyComponent, Stepper, DiagnosticComponent, TimeDifferencingWrapper, DataArray
)
import pytest
from numpy.testing import assert_allclose
from copy import deepcopy


class MockTendencyComponent(TendencyComponent):

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state):
        self._num_updates += 1
        return {}, {'num_updates': self._num_updates}


class MockStepper(Stepper):

    input_properties = {}

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
        super(MockStepper, self).__init__()

    def array_call(self, state, timestep):
        self._num_updates += 1

        return (
            {'num_updates': self._num_updates},
            {'value': 1}
        )


class MockStepperThatExpects(Stepper):

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


class MockTendencyComponentThatExpects(TendencyComponent):

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


class MockDiagnosticComponentThatExpects(DiagnosticComponent):

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
        self.implicit = MockStepper()
        self.wrapped = TimeDifferencingWrapper(self.implicit)
        self.state = {
            'time': timedelta(0),
            'value': DataArray([0], attrs={'units': 'm'})
        }

    def tearDown(self):
        self.component = None

    def testWrapperCallsImplicit(self):
        tendencies, diagnostics = self.wrapped(self.state, timedelta(seconds=1))
        assert diagnostics['num_updates'].values == 1
        tendencies, diagnostics = self.wrapped(self.state, timedelta(seconds=1))
        assert diagnostics['num_updates'].values == 2
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
            'time': timedelta(0),
            'value': DataArray([0.011], attrs={'units': 'km'})
        }
        tendencies, diagnostics = self.wrapped(state, timedelta(seconds=5))
        assert 'value' in tendencies.keys()
        assert isinstance(tendencies['value'], DataArray)
        assert_allclose(tendencies['value'].to_units('m s^-1').values[0], -2)
        assert_allclose(tendencies['value'].values[0], -2.)
        assert_allclose(tendencies['value'].to_units('km s^-1').values[0], -0.002)


if __name__ == '__main__':
    pytest.main([__file__])
