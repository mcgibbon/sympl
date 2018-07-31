from sympl._core.tracers import TracerPacker, reset_tracers, reset_packers
from sympl import (
    TendencyComponent, Stepper, DiagnosticComponent, ImplicitTendencyComponent, register_tracer,
    get_tracer_unit_dict, units_are_compatible, DataArray, InvalidPropertyDictError
)
import unittest
import numpy as np
import pytest
from datetime import timedelta


class MockTendencyComponent(TendencyComponent):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__(self, **kwargs):
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.tendency_properties = kwargs.pop('tendency_properties', {})
        self.diagnostic_output = {}
        self.tendency_output = {}
        self.times_called = 0
        self.state_given = None
        super(MockTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self.tendency_output, self.diagnostic_output


class MockTracerTendencyComponent(TendencyComponent):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    uses_tracers = True
    tracer_dims = ('tracer', '*')

    def __init__(self, **kwargs):
        prepend_tracers = kwargs.pop('prepend_tracers', None)
        if prepend_tracers is not None:
            self.prepend_tracers = prepend_tracers
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.tendency_properties = kwargs.pop('tendency_properties', {})
        self.diagnostic_output = {}
        self.times_called = 0
        self.state_given = None
        super(MockTracerTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return_state = {}
        return_state.update(state)
        return_state.pop('time')
        return return_state, self.diagnostic_output


class MockImplicitTendencyComponent(ImplicitTendencyComponent):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__( self, **kwargs):
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.tendency_properties = kwargs.pop('tendency_properties', {})
        self.diagnostic_output = {}
        self.tendency_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicitTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.tendency_output, self.diagnostic_output


class MockTracerImplicitTendencyComponent(ImplicitTendencyComponent):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    uses_tracers = True
    tracer_dims = ('tracer', '*')

    def __init__(self, **kwargs):
        prepend_tracers = kwargs.pop('prepend_tracers', None)
        if prepend_tracers is not None:
            self.prepend_tracers = prepend_tracers
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.tendency_properties = kwargs.pop('tendency_properties', {})
        self.diagnostic_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockTracerImplicitTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return_state = {}
        return_state.update(state)
        return_state.pop('time')
        return return_state, self.diagnostic_output


class MockDiagnosticComponent(DiagnosticComponent):

    input_properties = None
    diagnostic_properties = None

    def __init__(self, **kwargs):
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.diagnostic_output = {}
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

    def __init__(self, **kwargs):
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.output_properties = kwargs.pop('output_properties', {})
        self.diagnostic_output = {}
        self.state_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockStepper, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.diagnostic_output, self.state_output


class MockTracerStepper(Stepper):

    input_properties = None
    diagnostic_properties = None
    output_properties = None

    uses_tracers = True
    tracer_dims = ('tracer', '*')

    def __init__(self, **kwargs):
        prepend_tracers = kwargs.pop('prepend_tracers', None)
        if prepend_tracers is not None:
            self.prepend_tracers = prepend_tracers
        self.input_properties = kwargs.pop('input_properties', {})
        self.diagnostic_properties = kwargs.pop('diagnostic_properties', {})
        self.output_properties = kwargs.pop('output_properties', {})
        self.diagnostic_output = {}
        self.state_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockTracerStepper, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return_state = {}
        return_state.update(state)
        return_state.pop('time')
        return self.diagnostic_output, return_state


class RegisterTracerTests(unittest.TestCase):

    def setUp(self):
        reset_tracers()

    def tearDown(self):
        reset_tracers()

    def test_initially_empty(self):
        assert len(get_tracer_unit_dict()) == 0

    def test_register_one_tracer(self):
        register_tracer('tracer1', 'm')
        d = get_tracer_unit_dict()
        assert len(d) == 1
        assert 'tracer1' in d
        assert d['tracer1'] == 'm'

    def test_register_two_tracers(self):
        register_tracer('tracer1', 'm')
        register_tracer('tracer2', 'degK')
        d = get_tracer_unit_dict()
        assert len(d) == 2
        assert 'tracer1' in d
        assert 'tracer2' in d
        assert d['tracer1'] == 'm'
        assert d['tracer2'] == 'degK'

    def test_reregister_tracer(self):
        register_tracer('tracer1', 'm')
        register_tracer('tracer1', 'm')
        d = get_tracer_unit_dict()
        assert len(d) == 1
        assert 'tracer1' in d
        assert d['tracer1'] == 'm'

    def test_reregister_tracer_different_units(self):
        register_tracer('tracer1', 'm')
        with self.assertRaises(ValueError):
            register_tracer('tracer1', 'degK')


class TracerPackerBase(object):

    def setUp(self):
        reset_tracers()
        reset_packers()

    def tearDown(self):
        reset_tracers()
        reset_packers()

    def test_packs_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        packed = packer.pack({})
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (0, 0)

    def test_unpacks_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        unpacked = packer.unpack({}, {})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_packs_one_tracer(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        state = {'tracer1': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'g/m^3'})}
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == state['tracer1'].values)

    def test_packs_one_tracer_converts_units(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'kg/m^3')
        state = {'tracer1': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'g/m^3'})}
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == state['tracer1'].values * 1e-3)

    def test_packs_one_3d_tracer(self):
        np.random.seed(0)
        dims = ['tracer', 'latitude', 'longitude', 'mid_levels']
        register_tracer('tracer1', 'g/m^3')
        state = {
            'tracer1': DataArray(
                np.random.randn(2, 3, 4),
                dims=['latitude', 'longitude', 'mid_levels'],
                attrs={'units': 'g/m^3'}
            )
        }
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 2, 3, 4)
        assert np.all(packed[0, :, :, :] == state['tracer1'].values)

    def test_packer_does_not_change_input_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.component, dims)
        assert len(self.component.input_properties) == 0

    def test_packer_does_not_change_input_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        assert len(self.component.input_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert len(self.component.input_properties) == 0

    def test_packs_one_tracer_registered_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        state = {'tracer1': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'g/m^3'})}
        packer = TracerPacker(self.component, dims)
        register_tracer('tracer1', 'g/m^3')
        packed = packer.pack(state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == state['tracer1'].values)

    def test_packs_two_tracers(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg')
        state = {
            'tracer1': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'g/m^3'}),
            'tracer2': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'kg'})
        }
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 5)

    def test_packs_three_tracers_in_order_registered(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg'),
        register_tracer('tracer3', 'kg/m^3')
        state = {
            'tracer1': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'g/m^3'}),
            'tracer2': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'kg'}),
            'tracer3': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'kg/m^3'}),
        }
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (3, 5)
        assert np.all(packed[0, :] == state['tracer1'].values)
        assert np.all(packed[1, :] == state['tracer2'].values)
        assert np.all(packed[2, :] == state['tracer3'].values)

    def test_unpacks_three_tracers_in_order_registered(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg'),
        register_tracer('tracer3', 'kg/m^3')
        state = {
            'tracer1': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'g/m^3'}),
            'tracer2': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'kg'}),
            'tracer3': DataArray(np.random.randn(5), dims=['dim1'], attrs={'units': 'kg/m^3'}),
        }
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(state)
        unpacked = packer.unpack(packed, state)
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 3
        assert np.all(unpacked['tracer1'] == state['tracer1'])
        assert np.all(unpacked['tracer2'] == state['tracer2'])
        assert np.all(unpacked['tracer3'] == state['tracer3'])

    def test_packer_allows_overlap_input_registered_after_init(self):
        self.component = self.component.__class__(
            input_properties={
                'name': {
                    'units': 'm',
                    'dims': ['*'],
                }
            }
        )
        packer = TracerPacker(self.component, ['tracer', '*'])
        register_tracer('name', 'm')

    def test_packer_allows_overlap_input_registered_before_init(self):
        self.component = self.component.__class__(
            input_properties={
                'name': {
                    'units': 'm',
                    'dims': ['*'],
                }
            }
        )
        register_tracer('name', 'm')
        packer = TracerPacker(self.component, ['tracer', '*'])


class PrognosticTracerPackerTests(TracerPackerBase, unittest.TestCase):

    def setUp(self):
        self.component = MockTendencyComponent()
        super(PrognosticTracerPackerTests, self).setUp()

    def tearDown(self):
        self.component = None
        super(PrognosticTracerPackerTests, self).tearDown()

    def test_packer_does_not_change_tendency_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.component, dims)
        assert 'tracer1' not in self.component.tendency_properties
        assert len(self.component.tendency_properties) == 0

    def test_packer_does_not_change_tendency_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        assert len(self.component.tendency_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert len(self.component.tendency_properties) == 0

    def test_packer_wont_overwrite_tendency_registered_after_init(self):
        self.component = MockTendencyComponent(
            tendency_properties={
                'name': {
                    'units': 'm',
                    'dims': ['*'],
                }
            }
        )
        packer = TracerPacker(self.component, ['tracer', '*'])
        with self.assertRaises(InvalidPropertyDictError):
            register_tracer('name', 'm')

    def test_packer_wont_overwrite_tendency_registered_before_init(self):
        self.component = MockTendencyComponent(
            tendency_properties={
                'name': {
                    'units': 'm',
                    'dims': ['*'],
                }
            }
        )
        register_tracer('name', 'm')
        with self.assertRaises(InvalidPropertyDictError):
            packer = TracerPacker(self.component, ['tracer', '*'])


class ImplicitPrognosticTracerPackerTests(PrognosticTracerPackerTests):

    def setUp(self):
        self.component = MockImplicitTendencyComponent()
        super(ImplicitPrognosticTracerPackerTests, self).setUp()

    def tearDown(self):
        self.component = None
        super(ImplicitPrognosticTracerPackerTests, self).tearDown()


class ImplicitTracerPackerTests(TracerPackerBase, unittest.TestCase):

    def setUp(self):
        self.component = MockStepper()
        super(ImplicitTracerPackerTests, self).setUp()

    def tearDown(self):
        self.component = None
        super(ImplicitTracerPackerTests, self).tearDown()

    def test_packer_does_not_change_output_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.component, dims)
        assert len(self.component.output_properties) == 0

    def test_packer_does_not_change_output_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        assert len(self.component.output_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert len(self.component.output_properties) == 0

    def test_packer_wont_overwrite_output_registered_after_init(self):
        self.component = MockStepper(
            output_properties={
                'name': {
                    'units': 'm',
                    'dims': ['*'],
                }
            }
        )
        packer = TracerPacker(self.component, ['tracer', '*'])
        with self.assertRaises(InvalidPropertyDictError):
            register_tracer('name', 'm')

    def test_packer_wont_overwrite_output_registered_before_init(self):
        self.component = MockStepper(
            output_properties={
                'name': {
                    'units': 'm',
                    'dims': ['*'],
                }
            }
        )
        register_tracer('name', 'm')
        with self.assertRaises(InvalidPropertyDictError):
            packer = TracerPacker(self.component, ['tracer', '*'])


class DiagnosticTracerPackerTests(unittest.TestCase):

    def test_raises_on_diagnostic_init(self):
        diagnostic = MockDiagnosticComponent()
        with self.assertRaises(TypeError):
            TracerPacker(diagnostic, ['tracer', '*'])


class TracerComponentBase(object):

    def setUp(self):
        reset_tracers()
        reset_packers()

    def tearDown(self):
        reset_tracers()
        reset_packers()

    def call_component(self, input_state):
        pass

    def test_packs_no_tracers(self):
        input_state = {
            'time': timedelta(0),
        }
        self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (0, 0)

    def test_unpacks_no_tracers(self):
        input_state = {
            'time': timedelta(0),
        }
        unpacked = self.call_component(input_state)
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_packs_one_tracer(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            )
        }
        self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == input_state['tracer1'].values)

    def test_packs_one_prepended_tracer(self):
        np.random.seed(0)
        self.component = self.component.__class__(prepend_tracers=[('tracer1', 'g/m^3')])
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            )
        }
        self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == input_state['tracer1'].values)

    def test_packs_two_prepended_tracers(self):
        np.random.seed(0)
        self.component = self.component.__class__(
            prepend_tracers=[('tracer1', 'g/m^3'), ('tracer2', 'J/m^3')])
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            ),
            'tracer2': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'J/m^3'},
            ),
        }
        unpacked = self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 5)
        assert np.all(packed[0, :] == input_state['tracer1'].values)
        assert np.all(packed[1, :] == input_state['tracer2'].values)
        assert len(unpacked) == 2
        assert np.all(unpacked['tracer1'].values == input_state['tracer1'].values)
        assert np.all(unpacked['tracer2'].values == input_state['tracer2'].values)

    def test_packs_prepended_and_normal_tracers_register_first(self):
        register_tracer('tracer2', 'J/m^3')
        self.component = self.component.__class__(
            prepend_tracers=[('tracer1', 'g/m^3')])
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            ),
            'tracer2': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'J/m^3'},
            ),
        }
        unpacked = self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 5)
        assert np.all(packed[0, :] == input_state['tracer1'].values)
        assert np.all(packed[1, :] == input_state['tracer2'].values)
        assert len(unpacked) == 2
        assert np.all(
            unpacked['tracer1'].values == input_state['tracer1'].values)
        assert np.all(
            unpacked['tracer2'].values == input_state['tracer2'].values)

    def test_packs_prepended_and_normal_tracers_register_after_init(self):
        self.component = self.component.__class__(
            prepend_tracers=[('tracer1', 'g/m^3')])
        register_tracer('tracer2', 'J/m^3')
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            ),
            'tracer2': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'J/m^3'},
            ),
        }
        unpacked = self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 5)
        assert np.all(packed[0, :] == input_state['tracer1'].values)
        assert np.all(packed[1, :] == input_state['tracer2'].values)
        assert len(unpacked) == 2
        assert np.all(
            unpacked['tracer1'].values == input_state['tracer1'].values)
        assert np.all(
            unpacked['tracer2'].values == input_state['tracer2'].values)

    def test_packs_one_3d_tracer(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(1, 2, 3),
                dims=['dim1', 'dim2', 'dim3'],
                attrs={'units': 'g/m^3'},
            )
        }
        self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 6)
        assert np.all(packed[0, :] == input_state['tracer1'].values.flatten())

    def test_restores_one_3d_tracer(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(1, 2, 3),
                dims=['dim1', 'dim2', 'dim3'],
                attrs={'units': 'g/m^3'},
            )
        }
        unpacked = self.call_component(input_state)
        assert len(unpacked) == 1
        assert unpacked['tracer1'].shape == input_state['tracer1'].shape
        assert np.all(unpacked['tracer1'].values == input_state['tracer1'].values)

    def test_does_not_change_input_properties(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        assert 'tracer1' not in self.component.input_properties

    def test_packs_two_tracers(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg')
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            ),
            'tracer2': DataArray(
                np.random.randn(5),
                dims=['dim1'],
                attrs={'units': 'kg'}
            ),
        }
        unpacked = self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 5)
        assert np.all(packed[0, :] == input_state['tracer1'].values)
        assert np.all(packed[1, :] == input_state['tracer2'].values)
        assert len(unpacked) == 2
        assert np.all(unpacked['tracer1'].values == input_state['tracer1'].values)
        assert np.all(unpacked['tracer2'].values == input_state['tracer2'].values)

    def test_packing_differing_dims(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg')
        input_state = {
            'time': timedelta(0),
            'tracer1': DataArray(
                np.random.randn(2),
                dims=['dim1'],
                attrs={'units': 'g/m^3'},
            ),
            'tracer2': DataArray(
                np.random.randn(2),
                dims=['dim2'],
                attrs={'units': 'kg'}
            ),
        }
        unpacked = self.call_component(input_state)
        packed = self.component.state_given['tracers']
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 4)
        assert len(unpacked) == 2


class PrognosticTracerComponentTests(TracerComponentBase, unittest.TestCase):

    def setUp(self):
        super(PrognosticTracerComponentTests, self).setUp()
        self.component = MockTracerTendencyComponent()

    def tearDown(self):
        super(PrognosticTracerComponentTests, self).tearDown()
        self.component = None

    def call_component(self, input_state):
        return self.component(input_state)[0]


class ImplicitPrognosticTracerComponentTests(TracerComponentBase, unittest.TestCase):

    def setUp(self):
        super(ImplicitPrognosticTracerComponentTests, self).setUp()
        self.component = MockTracerImplicitTendencyComponent()

    def tearDown(self):
        super(ImplicitPrognosticTracerComponentTests, self).tearDown()
        self.component = None

    def call_component(self, input_state):
        return self.component(input_state, timedelta(hours=1))[0]


class ImplicitTracerComponentTests(TracerComponentBase, unittest.TestCase):

    def setUp(self):
        super(ImplicitTracerComponentTests, self).setUp()
        self.component = MockTracerStepper()

    def tearDown(self):
        super(ImplicitTracerComponentTests, self).tearDown()
        self.component = None

    def call_component(self, input_state):
        return self.component(input_state, timedelta(hours=1))[1]
