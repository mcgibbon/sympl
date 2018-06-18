from sympl._core.tracers import TracerPacker, clear_tracer_unit_dict
from sympl import Prognostic, register_tracer, get_tracer_unit_dict
import unittest
import numpy as np
import pytest


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
        clear_tracer_unit_dict()

    def test_initially_empty(self):
        assert len(get_tracer_unit_dict()) == 0

    def test_register_one_tracer(self):
        register_tracer('tracer1', 'm')
        d = get_tracer_unit_dict()
        assert len(d) == 1
        assert 'tracer1' in d
        assert d['tracer1'] == 'unit1'

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


class TracerPackerTests(unittest.TestCase):

    def setUp(self):
        self.prognostic = MockPrognostic()

    def tearDown(self):
        self.diagnostic = None

    def test_packs_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.prognostic, dims)
        packed = packer.pack({})
        assert isinstance(packed, np.ndarray)
        assert packed.shape == [0, 0]

    def test_unpacks_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.prognostic, dims)
        unpacked = packer.unpack({})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_unpacks_no_tracers_with_arrays_input(self):
        dims = ['tracer', '*']
        packer = TracerPacker(self.prognostic, dims)
        unpacked = packer.unpack({'air_temperature': np.zeros((5,))})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_packs_one_tracer(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        raw_state = {'tracer1': np.random.randn(5)}
        packer = TracerPacker(self.prognostic, dims)
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == [1, 5]
        assert np.all(packed[0, :] == raw_state['tracer1'])

    def test_packs_updates_properties(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        packer = TracerPacker(self.prognostic, dims)
        assert 'tracer1' in self.prognostic.input_properties
        assert tuple(self.prognostic.input_propertes['tracer1']['dims']) == ('*',)
        assert self.prognostic.input_properties['units'] == 'g/m^3'
        assert len(self.prognostic.input_properties) == 1

    def test_packs_updates_properties_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        packer = TracerPacker(self.prognostic, dims)
        assert len(self.prognostic.input_properties) == 0
        register_tracer('tracer1', 'g/m^3')
        assert 'tracer1' in self.prognostic.input_properties
        assert tuple(
            self.prognostic.input_propertes['tracer1']['dims']) == ('*',)
        assert self.prognostic.input_properties['units'] == 'g/m^3'
        assert len(self.prognostic.input_properties) == 1

    def test_packs_one_tracer_registered_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        raw_state = {'tracer1': np.random.randn(5)}
        packer = TracerPacker(self.prognostic, dims)
        register_tracer('tracer1', 'g/m^3')
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == [1, 5]
        assert np.all(packed[0, :] == raw_state['tracer1'])

