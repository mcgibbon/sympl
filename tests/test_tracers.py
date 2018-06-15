from sympl._core.tracers import TracerPacker, register_tracer
import unittest
import numpy as np

class TracerPackerTests(unittest.TestCase):

    def test_packs_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(dims)
        packed = packer.pack({})
        assert isinstance(packed, np.ndarray)
        assert packed.shape == [0, 0]

    def test_unpacks_no_tracers(self):
        dims = ['tracer', '*']
        packer = TracerPacker(dims)
        unpacked = packer.unpack({})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_unpacks_no_tracers_with_arrays_input(self):
        dims = ['tracer', '*']
        packer = TracerPacker(dims)
        unpacked = packer.unpack({'air_temperature': np.zeros((5,))})
        assert isinstance(unpacked, dict)
        assert len(unpacked) == 0

    def test_packs_one_tracer(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        register_tracer('tracer1', 'g/m^3')
        raw_state = {'tracer1': np.random.randn(5)}
        packer = TracerPacker(dims, raw_state)
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == [1, 5]
        assert np.all(packed[0, :] == raw_state['tracer1'])

    def test_packs_one_tracer_registered_after_init(self):
        np.random.seed(0)
        dims = ['tracer', '*']
        raw_state = {'tracer1': np.random.randn(5)}
        packer = TracerPacker(dims, raw_state)
        register_tracer('tracer1', 'g/m^3')
        packed = packer.pack(raw_state)
        assert isinstance(packed, np.ndarray)
        assert packed.shape == [1, 5]
        assert np.all(packed[0, :] == raw_state['tracer1'])

