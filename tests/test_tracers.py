from sympl._core.tracers import TracerPacker, clear_tracers, clear_packers
from sympl import (
    Prognostic, Implicit, Diagnostic, ImplicitPrognostic, register_tracer,
    get_tracer_unit_dict, units_are_compatible, DataArray
)
import unittest
import numpy as np
import pytest
from datetime import timedelta


class MockPrognostic(Prognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__(self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.tendency_properties = {}
        self.diagnostic_output = {}
        self.tendency_output = {}
        self.times_called = 0
        self.state_given = None
        super(MockPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self.tendency_output, self.diagnostic_output


class MockTracerPrognostic(Prognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    uses_tracers = True
    tracer_dims = ('tracer', '*')

    def __init__(self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.tendency_properties = {}
        self.diagnostic_output = {}
        self.times_called = 0
        self.state_given = None
        super(MockTracerPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return_state = {}
        return_state.update(state)
        return_state.pop('time')
        return return_state, self.diagnostic_output


class MockImplicitPrognostic(ImplicitPrognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__( self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.tendency_properties = {}
        self.diagnostic_output = {}
        self.tendency_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicitPrognostic, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.tendency_output, self.diagnostic_output


class MockTracerImplicitPrognostic(ImplicitPrognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    uses_tracers = True
    tracer_dims = ('tracer', '*')

    def __init__( self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.tendency_properties = {}
        self.diagnostic_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockTracerImplicitPrognostic, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return_state = {}
        return_state.update(state)
        return_state.pop('time')
        return return_state, self.diagnostic_output


class MockDiagnostic(Diagnostic):

    input_properties = None
    diagnostic_properties = None

    def __init__(self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.diagnostic_output = {}
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

    def __init__(self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.output_properties = {}
        self.diagnostic_output = {}
        self.state_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockImplicit, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return self.diagnostic_output, self.state_output


class MockTracerImplicit(Implicit):

    input_properties = None
    diagnostic_properties = None
    output_properties = None

    uses_tracers = True
    tracer_dims = ('tracer', '*')

    def __init__(self, **kwargs):
        self.input_properties = {}
        self.diagnostic_properties = {}
        self.output_properties = {}
        self.diagnostic_output = {}
        self.state_output = {}
        self.times_called = 0
        self.state_given = None
        self.timestep_given = None
        super(MockTracerImplicit, self).__init__(**kwargs)

    def array_call(self, state, timestep):
        self.times_called += 1
        self.state_given = state
        self.timestep_given = timestep
        return_state = {}
        return_state.update(state)
        return_state.pop('time')
        return self.diagnostic_output, return_state

"""
On init and tracer registration, should update input properties of its
component.

On pack, should pack all tracers not already present in input_properties.

On unpack, should unpack all tracers not present in input_properties

Keep track of internal representation of tracer order.
    On adding new tracer, add it to the end.
    Allow global tracer order to be set.
"""


class RegisterTracerTests(unittest.TestCase):

    def setUp(self):
        clear_tracers()

    def tearDown(self):
        clear_tracers()

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
        clear_tracers()
        clear_packers()

    def tearDown(self):
        clear_tracers()
        clear_packers()

    def test_packs_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        packed = packer.pack({})
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (0, 0)

    def test_unpacks_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        unpacked = packer.unpack({})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_unpacks_no_tracers_with_arrays_input(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        unpacked = packer.unpack({'air_temperature': np.zeros((5,))})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_packs_one_tracer(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        raw_state = {'tracer1': np.random.randn(5)}
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == raw_state['tracer1'])

    def test_packs_one_3d_tracer(self):
        np.random.seed(0)
        dims = ['tracer', 'latitude', 'longitude', 'mid_levels']
        register_tracer('tracer1', 'g/m^3')
        raw_state = {'tracer1': np.random.randn(2, 3, 4)}
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 2, 3, 4)
        assert np.all(packed[0, :, :, :] == raw_state['tracer1'])

    def test_packs_updates_input_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.component, dims)
        assert 'tracer1' in self.component.input_properties
        assert tuple(self.component.input_properties['tracer1']['dims']) == ('*',)
        assert self.component.input_properties['tracer1']['units'] == 'g/m^3'
        assert len(self.component.input_properties) == 1

    def test_packs_updates_input_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        assert len(self.component.input_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert 'tracer1' in self.component.input_properties
        assert tuple(
            self.component.input_properties['tracer1']['dims']) == ('*',)
        assert self.component.input_properties['tracer1']['units'] == 'g/m^3'
        assert len(self.component.input_properties) == 1

    def test_packs_one_tracer_registered_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        raw_state = {'tracer1': np.random.randn(5)}
        packer = TracerPacker(self.component, dims)
        register_tracer('tracer1', 'g/m^3')
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (1, 5)
        assert np.all(packed[0, :] == raw_state['tracer1'])

    def test_packs_two_tracers(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg')
        raw_state = {'tracer1': np.random.randn(5), 'tracer2': np.random.randn(5)}
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (2, 5)

    def test_packs_three_tracers_in_order_registered(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg'),
        register_tracer('tracer3', 'kg/m^3')
        raw_state = {
            'tracer1': np.random.randn(5),
            'tracer2': np.random.randn(5),
            'tracer3': np.random.randn(5),
        }
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == (3, 5)
        assert np.all(packed[0, :] == raw_state['tracer1'])
        assert np.all(packed[1, :] == raw_state['tracer2'])
        assert np.all(packed[2, :] == raw_state['tracer3'])

    def test_unpacks_three_tracers_in_order_registered(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        register_tracer('tracer2', 'kg'),
        register_tracer('tracer3', 'kg/m^3')
        raw_state = {
            'tracer1': np.random.randn(5),
            'tracer2': np.random.randn(5),
            'tracer3': np.random.randn(5),
        }
        packer = TracerPacker(self.component, dims)
        packed = packer.pack(raw_state)
        unpacked = packer.unpack(packed)
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 3
        assert np.all(unpacked['tracer1'] == raw_state['tracer1'])
        assert np.all(unpacked['tracer2'] == raw_state['tracer2'])
        assert np.all(unpacked['tracer3'] == raw_state['tracer3'])


class PrognosticTracerPackerTests(TracerPackerBase, unittest.TestCase):

    def setUp(self):
        self.component = MockPrognostic()
        super(PrognosticTracerPackerTests, self).setUp()

    def tearDown(self):
        self.component = None
        super(PrognosticTracerPackerTests, self).tearDown()

    def test_packs_updates_tendency_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.component, dims)
        assert 'tracer1' in self.component.tendency_properties
        assert tuple(self.component.tendency_properties['tracer1']['dims']) == ('*',)
        assert units_are_compatible(self.component.tendency_properties['tracer1']['units'], 'g/m^3 s^-1')
        assert len(self.component.tendency_properties) == 1

    def test_packs_updates_tendency_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        assert len(self.component.tendency_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert 'tracer1' in self.component.tendency_properties
        assert tuple(
            self.component.tendency_properties['tracer1']['dims']) == ('*',)
        assert units_are_compatible(self.component.tendency_properties['tracer1']['units'], 'g/m^3 s^-1')
        assert len(self.component.tendency_properties) == 1


class ImplicitPrognosticTracerPackerTests(PrognosticTracerPackerTests):

    def setUp(self):
        self.component = MockImplicitPrognostic()
        super(ImplicitPrognosticTracerPackerTests, self).setUp()

    def tearDown(self):
        self.component = None
        super(ImplicitPrognosticTracerPackerTests, self).tearDown()


class ImplicitTracerPackerTests(TracerPackerBase, unittest.TestCase):

    def setUp(self):
        self.component = MockImplicit()
        super(ImplicitTracerPackerTests, self).setUp()

    def tearDown(self):
        self.component = None
        super(ImplicitTracerPackerTests, self).tearDown()

    def test_packs_updates_output_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.component, dims)
        assert 'tracer1' in self.component.output_properties
        assert tuple(self.component.output_properties['tracer1']['dims']) == ('*',)
        assert self.component.output_properties['tracer1']['units'] == 'g/m^3'
        assert len(self.component.output_properties) == 1

    def test_packs_updates_output_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.component, dims)
        assert len(self.component.output_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert 'tracer1' in self.component.output_properties
        assert tuple(
            self.component.output_properties['tracer1']['dims']) == ('*',)
        assert self.component.output_properties['tracer1']['units'] == 'g/m^3'
        assert len(self.component.output_properties) == 1


class DiagnosticTracerPackerTests(unittest.TestCase):

    def test_raises_on_diagnostic_init(self):
        diagnostic = MockDiagnostic()
        with self.assertRaises(TypeError):
            TracerPacker(diagnostic, ['tracer', '*'])


class TracerComponentBase(object):

    def setUp(self):
        clear_tracers()
        clear_packers()

    def tearDown(self):
        clear_tracers()
        clear_packers()

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

    def test_updates_input_properties(self):
        np.random.seed(0)
        register_tracer('tracer1', 'g/m^3')
        assert 'tracer1' in self.component.input_properties
        assert tuple(self.component.input_properties['tracer1']['dims']) == ('*',)
        assert self.component.input_properties['tracer1']['units'] == 'g/m^3'
        assert len(self.component.input_properties) == 1

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
        self.component = MockTracerPrognostic()

    def tearDown(self):
        super(PrognosticTracerComponentTests, self).tearDown()
        self.component = None

    def call_component(self, input_state):
        return self.component(input_state)[0]


class ImplicitPrognosticTracerComponentTests(TracerComponentBase, unittest.TestCase):

    def setUp(self):
        super(ImplicitPrognosticTracerComponentTests, self).setUp()
        self.component = MockTracerImplicitPrognostic()

    def tearDown(self):
        super(ImplicitPrognosticTracerComponentTests, self).tearDown()
        self.component = None

    def call_component(self, input_state):
        return self.component(input_state, timedelta(hours=1))[0]


class ImplicitTracerComponentTests(TracerComponentBase, unittest.TestCase):

    def setUp(self):
        super(ImplicitTracerComponentTests, self).setUp()
        self.component = MockTracerImplicit()

    def tearDown(self):
        super(ImplicitTracerComponentTests, self).tearDown()
        self.component = None

    def call_component(self, input_state):
        return self.component(input_state, timedelta(hours=1))[1]
