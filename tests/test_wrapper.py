from datetime import timedelta, datetime
import unittest
from sympl import (
    TendencyComponent, Stepper, DiagnosticComponent, UpdateFrequencyWrapper, ScalingWrapper,
    TimeDifferencingWrapper, DataArray, ImplicitTendencyComponent
)
import pytest
import numpy as np


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
        self.diagnostic_output = diagnostic_output
        self.tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        super(MockTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self.tendency_output, self.diagnostic_output


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
        self.diagnostic_output = diagnostic_output
        self.tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicitTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.tendency_output, self.diagnostic_output


class MockDiagnosticComponent(DiagnosticComponent):

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
        super(MockDiagnosticComponent, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self.diagnostic_output


class MockStepper(Stepper):

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
        super(MockStepper, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.diagnostic_output, self.state_output


class MockEmptyPrognostic(MockTendencyComponent):

    def __init__(self, **kwargs):
        super(MockEmptyPrognostic, self).__init__(
            input_properties={},
            diagnostic_properties={},
            tendency_properties={},
            diagnostic_output={},
            tendency_output={},
            **kwargs
        )


class MockEmptyImplicitPrognostic(MockImplicitTendencyComponent):
    def __init__(self, **kwargs):
        super(MockEmptyImplicitPrognostic, self).__init__(
            input_properties={},
            diagnostic_properties={},
            tendency_properties={},
            diagnostic_output={},
            tendency_output={},
            **kwargs
        )


class MockEmptyDiagnostic(MockDiagnosticComponent):

    def __init__(self, **kwargs):
        super(MockEmptyDiagnostic, self).__init__(
            input_properties={},
            diagnostic_properties={},
            diagnostic_output={},
            **kwargs
        )


class MockEmptyImplicit(MockStepper):

    def __init__(self, **kwargs):
        super(MockEmptyImplicit, self).__init__(
            input_properties={},
            diagnostic_properties={},
            output_properties={},
            diagnostic_output={},
            state_output={},
            **kwargs
        )


class UpdateFrequencyBase(object):

    def get_component(self):
        raise NotImplementedError()

    def call_component(self, component, state):
        raise NotImplementedError()

    def test_set_update_frequency_calls_initially(self):
        component = UpdateFrequencyWrapper(self.get_component(), timedelta(hours=1))
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(hours=0)}
        result = self.call_component(component, state)
        assert component.times_called == 1

    def test_set_update_frequency_does_not_repeat_call_at_same_timedelta(self):
        component = UpdateFrequencyWrapper(self.get_component(), timedelta(hours=1))
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(hours=0)}
        result = self.call_component(component, state)
        result = self.call_component(component, state)
        assert component.times_called == 1

    def test_set_update_frequency_does_not_repeat_call_at_same_datetime(self):
        component = UpdateFrequencyWrapper(self.get_component(), timedelta(hours=1))
        assert isinstance(component, self.component_type)
        state = {'time': datetime(2010, 1, 1)}
        result = self.call_component(component, state)
        result = self.call_component(component, state)
        assert component.times_called == 1

    def test_set_update_frequency_updates_result_when_equal(self):
        component = UpdateFrequencyWrapper(self.get_component(), timedelta(hours=1))
        assert isinstance(component, self.component_type)
        result = self.call_component(component, {'time': timedelta(hours=0)})
        result = self.call_component(component, {'time': timedelta(hours=1)})
        assert component.times_called == 2

    def test_set_update_frequency_updates_result_when_greater(self):
        component = UpdateFrequencyWrapper(self.get_component(), timedelta(hours=1))
        assert isinstance(component, self.component_type)
        result = self.call_component(component, {'time': timedelta(hours=0)})
        result = self.call_component(component, {'time': timedelta(hours=2)})
        assert component.times_called == 2

    def test_set_update_frequency_does_not_update_when_less(self):
        component = UpdateFrequencyWrapper(self.get_component(), timedelta(hours=1))
        assert isinstance(component, self.component_type)
        result = self.call_component(component, {'time': timedelta(hours=0)})
        result = self.call_component(component, {'time': timedelta(minutes=59)})
        assert component.times_called == 1


class PrognosticUpdateFrequencyTests(unittest.TestCase, UpdateFrequencyBase):

    component_type = TendencyComponent

    def get_component(self):
        return MockEmptyPrognostic()

    def call_component(self, component, state):
        return component(state)


class ImplicitPrognosticUpdateFrequencyTests(unittest.TestCase, UpdateFrequencyBase):

    component_type = ImplicitTendencyComponent

    def get_component(self):
        return MockEmptyImplicitPrognostic()

    def call_component(self, component, state):
        return component(state, timestep=timedelta(hours=1))


class ImplicitUpdateFrequencyTests(unittest.TestCase, UpdateFrequencyBase):

    component_type = Stepper

    def get_component(self):
        return MockEmptyImplicit()

    def call_component(self, component, state):
        return component(state, timedelta(minutes=1))


class DiagnosticUpdateFrequencyTests(unittest.TestCase, UpdateFrequencyBase):

    component_type = DiagnosticComponent

    def get_component(self):
        return MockEmptyDiagnostic()

    def call_component(self, component, state):
        return component(state)


def test_scaled_component_wrong_type():
    class WrongObject(object):
        def __init__(self):
            self.a = 1

    wrong_component = WrongObject()

    with pytest.raises(TypeError):
        ScalingWrapper(wrong_component)


class ScalingInputMixin(object):

    def test_inputs_no_scaling(self):
        self.input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
        }
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        base_component = self.get_component()
        component = ScalingWrapper(base_component, input_scale_factors={})
        assert isinstance(component, self.component_type)
        self.call_component(component, state)
        assert base_component.state_given.keys() == state.keys()
        assert np.all(base_component.state_given['input1'] == state['input1'].values)

    def test_inputs_one_scaling(self):
        self.input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
            'input2': {
                'dims': ['dim1'],
                'units': 'm',
            },
        }
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
            'input2': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            input_scale_factors={
                'input1': 10.
            })
        assert isinstance(component, self.component_type)
        self.call_component(component, state)
        assert base_component.state_given.keys() == state.keys()
        assert np.all(base_component.state_given['input1'] == state['input1'].values * 10.)
        assert np.all(base_component.state_given['input2'] == state['input2'].values)

    def test_inputs_two_scalings(self):
        self.input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
            'input2': {
                'dims': ['dim1'],
                'units': 'm',
            },
        }
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
            'input2': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            input_scale_factors={
                'input1': 10.,
                'input2': 5.,
            })
        assert isinstance(component, self.component_type)
        self.call_component(component, state)
        assert base_component.state_given.keys() == state.keys()
        assert np.all(base_component.state_given['input1'] == 10.)
        assert np.all(base_component.state_given['input2'] == 5.)

    def test_inputs_one_scaling_with_unit_conversion(self):
        self.input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
            'input2': {
                'dims': ['dim1'],
                'units': 'm',
            },
        }
        state = {
            'time': timedelta(0),
            'input1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'km'}
            ),
            'input2': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            ),
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            input_scale_factors={
                'input1': 0.5
            })
        assert isinstance(component, self.component_type)
        self.call_component(component, state)
        assert base_component.state_given.keys() == state.keys()
        assert np.all(base_component.state_given['input1'] == 500.)
        assert np.all(base_component.state_given['input2'] == 1.)


class ScalingOutputMixin(object):

    def test_output_no_scaling(self):
        self.output_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.output_state = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            output_scale_factors={},
        )
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(0)}
        outputs = self.get_outputs(self.call_component(component, state))
        assert outputs.keys() == self.output_state.keys()
        assert np.all(outputs['diag1'] == 1.)

    def test_output_one_scaling(self):
        self.output_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.output_state = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            output_scale_factors={
                'diag1': 10.,
            },
        )
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(0)}
        outputs = self.get_outputs(
            self.call_component(component, state))
        assert outputs.keys() == self.output_state.keys()
        assert np.all(outputs['diag1'] == 10.)

    def test_output_no_scaling_when_input_scaled(self):
        self.input_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        self.output_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.output_state = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            input_scale_factors={
                'diag1': 10.,
            },
        )
        assert isinstance(component, self.component_type)
        state = {
            'time': timedelta(0),
            'diag1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        outputs = self.get_outputs(
            self.call_component(component, state))
        assert outputs.keys() == self.output_state.keys()
        assert np.all(outputs['diag1'] == 1.)


class ScalingDiagnosticMixin(object):

    def test_diagnostic_no_scaling(self):
        self.diagnostic_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.diagnostic_output = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            diagnostic_scale_factors={},
        )
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(0)}
        diagnostics = self.get_diagnostics(self.call_component(component, state))
        assert diagnostics.keys() == self.diagnostic_output.keys()
        assert np.all(diagnostics['diag1'] == 1.)

    def test_diagnostic_one_scaling(self):
        self.diagnostic_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.diagnostic_output = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            diagnostic_scale_factors={
                'diag1': 10.,
            },
        )
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(0)}
        diagnostics = self.get_diagnostics(
            self.call_component(component, state))
        assert diagnostics.keys() == self.diagnostic_output.keys()
        assert np.all(diagnostics['diag1'] == 10.)

    def test_diagnostic_no_scaling_when_input_scaled(self):
        self.input_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        self.diagnostic_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.diagnostic_output = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            input_scale_factors={
                'diag1': 10.,
            },
        )
        assert isinstance(component, self.component_type)
        state = {
            'time': timedelta(0),
            'diag1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        diagnostics = self.get_diagnostics(
            self.call_component(component, state))
        assert diagnostics.keys() == self.diagnostic_output.keys()
        assert np.all(diagnostics['diag1'] == 1.)


class ScalingTendencyMixin(object):

    def test_tendency_no_scaling(self):
        self.tendency_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.tendency_output = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            tendency_scale_factors={},
        )
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(0)}
        tendencies = self.get_tendencies(self.call_component(component, state))
        assert tendencies.keys() == self.tendency_output.keys()
        assert np.all(tendencies['diag1'] == 1.)

    def test_tendency_one_scaling(self):
        self.tendency_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        self.tendency_output = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            tendency_scale_factors={
                'diag1': 10.,
            },
        )
        assert isinstance(component, self.component_type)
        state = {'time': timedelta(0)}
        tendencies = self.get_tendencies(
            self.call_component(component, state))
        assert tendencies.keys() == self.tendency_output.keys()
        assert np.all(tendencies['diag1'] == 10.)

    def test_tendency_no_scaling_when_input_scaled(self):
        self.input_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        self.tendency_properties = {
            'diag1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        }
        self.tendency_output = {
            'diag1': np.ones([10])
        }
        base_component = self.get_component()
        component = ScalingWrapper(
            base_component,
            input_scale_factors={
                'diag1': 10.,
            },
        )
        assert isinstance(component, self.component_type)
        state = {
            'time': timedelta(0),
            'diag1': DataArray(
                np.ones([10]),
                dims=['dim1'],
                attrs={'units': 'm'}
            )
        }
        tendencies = self.get_tendencies(
            self.call_component(component, state))
        assert tendencies.keys() == self.tendency_output.keys()
        assert np.all(tendencies['diag1'] == 1.)


class DiagnosticScalingTests(
    unittest.TestCase, ScalingInputMixin, ScalingDiagnosticMixin):

    component_type = DiagnosticComponent

    def setUp(self):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.diagnostic_output = {}

    def get_component(self):
        return MockDiagnosticComponent(
            self.input_properties,
            self.diagnostic_properties,
            self.diagnostic_output
        )

    def get_diagnostics(self, output):
        return output

    def call_component(self, component, state):
        return component(state)


class PrognosticScalingTests(
    unittest.TestCase, ScalingInputMixin, ScalingDiagnosticMixin, ScalingTendencyMixin):

    component_type = TendencyComponent

    def setUp(self):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.tendency_properties = {}
        self.diagnostic_output = {}
        self.tendency_output = {}

    def get_component(self):
        return MockTendencyComponent(
            self.input_properties,
            self.diagnostic_properties,
            self.tendency_properties,
            self.diagnostic_output,
            self.tendency_output,
        )

    def get_diagnostics(self, output):
        return output[1]

    def get_tendencies(self, output):
        return output[0]

    def call_component(self, component, state):
        return component(state)


class ImplicitPrognosticScalingTests(
    unittest.TestCase, ScalingInputMixin, ScalingDiagnosticMixin,
    ScalingTendencyMixin):

    component_type = ImplicitTendencyComponent

    def setUp(self):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.tendency_properties = {}
        self.diagnostic_output = {}
        self.tendency_output = {}

    def get_component(self):
        return MockImplicitTendencyComponent(
            self.input_properties,
            self.diagnostic_properties,
            self.tendency_properties,
            self.diagnostic_output,
            self.tendency_output,
        )

    def get_diagnostics(self, output):
        return output[1]

    def get_tendencies(self, output):
        return output[0]

    def call_component(self, component, state):
        return component(state, timedelta(hours=1))


class ImplicitScalingTests(
    unittest.TestCase, ScalingInputMixin, ScalingDiagnosticMixin,
    ScalingOutputMixin):

    component_type = Stepper

    def setUp(self):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.output_properties = {}
        self.diagnostic_output = {}
        self.output_state = {}

    def get_component(self):
        return MockStepper(
            self.input_properties,
            self.diagnostic_properties,
            self.output_properties,
            self.diagnostic_output,
            self.output_state,
        )

    def get_diagnostics(self, output):
        return output[0]

    def get_outputs(self, output):
        return output[1]

    def call_component(self, component, state):
        return component(state, timedelta(hours=1))

if __name__ == '__main__':
    pytest.main([__file__])
