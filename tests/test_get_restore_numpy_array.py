import pytest
from sympl import (
    DataArray, get_numpy_array,
    restore_dimensions, get_numpy_arrays_with_properties,
    restore_data_arrays_with_properties, InvalidStateError,
    InvalidPropertyDictError)
import numpy as np
import unittest

try:
    from numpy.lib.array_utils import byte_bounds
except ImportError:
    from numpy import byte_bounds

def arrays_share_same_memory_space(source, target):
    
    if target.base is None:
        return target is source
    else:
        return target.base is source


def test_get_numpy_array_3d_no_change():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['x', 'y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['x', 'y', 'z'])
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert np.all(numpy_array == array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_3d_reverse():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['x', 'y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['z', 'y', 'x'])
    assert numpy_array.shape == (4, 3, 2)
    assert np.all(np.transpose(numpy_array, (2, 1, 0)) == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_2d_reverse():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['z', 'y'])
    assert numpy_array.shape == (3, 2)
    assert np.all(np.transpose(numpy_array, (1, 0)) == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_1d():
    array = DataArray(
        np.random.randn(2),
        dims=['y'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['y'])
    assert numpy_array.shape == (2,)
    assert np.all(numpy_array == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_creates_new_dim():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['x', 'y'])
    assert numpy_array.shape == (2, 1)
    assert np.all(numpy_array[:, 0] == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_creates_new_dim_in_front():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['y', 'x'])
    assert numpy_array.shape == (1, 2)
    assert np.all(numpy_array[0, :] == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_retrieves_explicit_dimensions():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['alpha', 'zeta'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['zeta', 'alpha'])
    assert numpy_array.shape == (3, 2)
    assert np.all(np.transpose(numpy_array, (1, 0)) == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_invalid_dimension_raises_value_error():
    array = DataArray(
        np.random.randn(2),
        dims=['sheep'],
        attrs={'units': ''},
    )
    try:
        numpy_array = get_numpy_array(array, ['y'])
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('Expected ValueError but no error was raised')


def test_get_numpy_array_invalid_dimension_collected_by_asterisk():
    array = DataArray(
        np.random.randn(2),
        dims=['sheep'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['*'])
    assert numpy_array.shape == (2,)
    assert np.all(numpy_array == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_dimension_not_listed_raises_value_error():
    array = DataArray(
        np.random.randn(2),
        dims=['z'],
        attrs={'units': ''},
    )
    try:
        numpy_array = get_numpy_array(array, ['y'])
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('Expected ValueError but no error was raised')


def test_get_numpy_array_no_dimensions_listed_raises_value_error():
    array = DataArray(
        np.random.randn(2),
        dims=['z'],
        attrs={'units': ''},
    )
    try:
        numpy_array = get_numpy_array(array, [])
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('Expected ValueError but no error was raised')


def test_get_numpy_array_not_enough_out_dims():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['x', 'y'],
        attrs={'units': ''},
    )
    try:
        numpy_array = get_numpy_array(array, ['x'])
    except ValueError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('Expected ValueError but no error was raised')


def test_get_numpy_array_asterisk_creates_new_dim():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['x', '*'])
    assert numpy_array.shape == (2, 1)
    assert np.all(numpy_array[:, 0] == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_asterisk_creates_new_dim_reversed():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['*', 'x'])
    assert numpy_array.shape == (1, 2)
    assert np.all(numpy_array[0, :] == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_asterisk_flattens():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['*'])
    assert numpy_array.shape == (6,)
    assert np.all(numpy_array.reshape((2, 3)) == array.values)
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_get_numpy_array_complicated_asterisk():
    array = DataArray(
        np.random.randn(2, 3, 4, 5),
        dims=['x', 'h', 'y', 'q'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'x', 'y'])
    for i in range(2):
        for j in range(4):
            assert np.allclose(numpy_array[:, i, j], array.values[i, :, j, :].flatten())
    # copying may take place in this case, so no more asserts


def test_get_numpy_array_zyx_to_starz_doesnt_copy():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    original_array = array.values
    numpy_array = get_numpy_array(array, ['*', 'z'])
    for i in range(2):
        assert np.all(numpy_array[:, i] == array.values[i, :, :].flatten())
    assert original_array is array.values
    assert byte_bounds(numpy_array) == byte_bounds(array.values)
    assert arrays_share_same_memory_space(array.values, numpy_array)


def test_restore_dimensions_complicated_asterisk():
    array = DataArray(
        np.random.randn(2, 3, 4, 5),
        dims=['x', 'h', 'y', 'q'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'x', 'y'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'x', 'y'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0


def test_restore_dimensions_transpose_alpha_beta():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['alpha', 'beta'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['beta', 'alpha'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['beta', 'alpha'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0


def test_restore_dimensions_starz_to_zyx():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'z'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0


def test_restore_dimensions_starz_to_zalphabeta():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'alpha', 'beta'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'z'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0


def test_restore_dimensions_starz_to_zyx():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'z'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0


def test_restore_dimensions_starz_to_zyx_has_no_attrs():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'z'], result_like=array)
    assert len(restored_array.attrs) == 0


def test_restore_dimensions_starz_to_zyx_doesnt_copy():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'z'], result_like=array)
    assert byte_bounds(restored_array.values) == byte_bounds(
        array.values)
    assert arrays_share_same_memory_space(
        array.values, restored_array.values)


def test_restore_dimensions_starz_to_zyx_with_attrs():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*', 'z'], result_like=array, result_attrs={'units': 'K'})
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 1
    assert 'units' in restored_array.attrs
    assert restored_array.attrs['units'] == 'K'


def test_restore_dimensions_3d_reverse():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['z', 'y', 'x'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['x', 'y', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['x', 'y', 'z'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0
    assert byte_bounds(restored_array.values) == byte_bounds(
        array.values)
    assert arrays_share_same_memory_space(
        array.values, restored_array.values)


def test_restore_dimensions_1d_flatten():
    array = DataArray(
        np.random.randn(2),
        dims=['z'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0
    assert byte_bounds(restored_array.values) == byte_bounds(
        array.values)
    assert arrays_share_same_memory_space(
        array.values, restored_array.values)


def test_restore_dimensions_2d_flatten():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['z', 'y'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['*'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['*'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0
    assert byte_bounds(restored_array.values) == byte_bounds(
        array.values)
    assert arrays_share_same_memory_space(
        array.values, restored_array.values)


def test_restore_dimensions_removes_dummy_axes():
    array = DataArray(
        np.random.randn(2),
        dims=['z'],
        attrs={'units': ''}
    )
    numpy_array = get_numpy_array(array, ['x', 'y', 'z'])
    restored_array = restore_dimensions(
        numpy_array, from_dims=['x', 'y', 'z'], result_like=array)
    assert np.all(restored_array.values == array.values)
    assert len(restored_array.attrs) == 0
    assert byte_bounds(restored_array.values) == byte_bounds(
        array.values)
    assert arrays_share_same_memory_space(
        array.values, restored_array.values)
    
class GetNumpyArraysWithPropertiesTests(unittest.TestCase):

    def test_returns_numpy_array(self):
        T_array = np.zeros([2, 3, 4], dtype=np.float64) + 280.
        property_dictionary = {
            'air_temperature': {
                'units': 'degK',
                'dims': ['x', 'y', 'z'],
            },
        }
        state = {
            'air_temperature': DataArray(
                T_array,
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature'], np.ndarray)
        assert byte_bounds(return_value['air_temperature']) == byte_bounds(
            T_array)
        assert arrays_share_same_memory_space(
            return_value['air_temperature'], T_array)

    def test_returns_numpy_array_using_alias(self):
        T_array = np.zeros([2, 3, 4], dtype=np.float64) + 280.
        property_dictionary = {
            'air_temperature': {
                'units': 'degK',
                'dims': ['x', 'y', 'z'],
                'alias': 'T',
            },
        }
        state = {
            'air_temperature': DataArray(
                T_array,
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['T'], np.ndarray)
        assert byte_bounds(return_value['T']) == byte_bounds(
            T_array)
        assert arrays_share_same_memory_space(
            return_value['T'], T_array)

    def test_returns_numpy_array_alias_doesnt_apply_to_state(self):
        T_array = np.zeros([2, 3, 4], dtype=np.float64) + 280.
        property_dictionary = {
            'air_temperature': {
                'units': 'degK',
                'dims': ['x', 'y', 'z'],
                'alias': 'T',
            },
        }
        state = {
            'T': DataArray(
                T_array,
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            ),
        }
        try:
            get_numpy_arrays_with_properties(
                state, property_dictionary)
        except KeyError:
            pass
        else:
            raise AssertionError('should have raised KeyError')

    def test_returns_scalar_array(self):
        T_array = np.array(0.)
        property_dictionary = {
            'air_temperature': {
                'units': 'degK',
                'dims': [],
            },
        }
        state = {
            'air_temperature': DataArray(
                T_array,
                dims=[],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature'], np.ndarray)
        assert byte_bounds(return_value['air_temperature']) == byte_bounds(
            T_array)

    def test_scalar_becomes_multidimensional_array(self):
        T_array = np.array(0.)
        property_dictionary = {
            'air_temperature': {
                'units': 'degK',
                'dims': ['z'],
            },
        }
        state = {
            'air_temperature': DataArray(
                T_array,
                dims=[],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature'], np.ndarray)
        assert len(return_value['air_temperature'].shape) == 1
        assert byte_bounds(return_value['air_temperature']) == byte_bounds(
            T_array)
        assert arrays_share_same_memory_space(
            T_array, return_value['air_temperature'])

    def test_creates_length_1_dimensions(self):
        T_array = np.zeros([4], dtype=np.float64) + 280.
        property_dictionary = {
            'air_temperature': {
                'dims': ['*', 'z'],
                'units': 'degK',
            },
        }
        state = {
            'air_temperature': DataArray(
                T_array,
                dims=['z'],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature'], np.ndarray)
        assert byte_bounds(
            return_value['air_temperature']) == byte_bounds(
            T_array)
        assert arrays_share_same_memory_space(
            T_array, return_value['air_temperature'])
        assert return_value['air_temperature'].shape == (1, 4)

    def test_only_requested_properties_are_returned(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
            'air_pressure': DataArray(
                np.zeros([2,2,4], dtype=np.float64),
                dims=['x', 'y', 'z'],
                attrs={'units': 'Pa'},
            )
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert 'air_temperature' in return_value.keys()

    def test_all_requested_properties_are_returned(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            },
            'air_pressure': {
                'dims': ['x', 'y', 'z'],
                'units': 'Pa',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
            'air_pressure': DataArray(
                np.zeros([2,2,4], dtype=np.float64),
                dims=['x', 'y', 'z'],
                attrs={'units': 'Pa'},
            ),
            'eastward_wind': DataArray(
                np.zeros([2,2,4], dtype=np.float64),
                attrs={'units': 'm/s'}
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 2
        assert 'air_temperature' in return_value.keys()
        assert 'air_pressure' in return_value.keys()
        assert np.all(return_value['air_temperature'] == 0.)
        assert np.all(return_value['air_pressure'] == 0.)

    def test_converts_units(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degC',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert 'air_temperature' in return_value.keys()
        assert np.all(return_value['air_temperature'] == -273.15)

    def test_unit_conversion_doesnt_modify_input(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degC',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert np.all(state['air_temperature'].values == 0.)
        assert state['air_temperature'].attrs['units'] is 'degK'

    def test_converting_units_maintains_float32_dtype(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degC',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float32),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
        }
        return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert 'air_temperature' in return_value.keys()
        assert return_value['air_temperature'].dtype is np.dtype('float32')

    def test_raises_if_units_property_undefined(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
        }
        try:
            return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        except InvalidPropertyDictError:
            pass
        else:
            raise AssertionError('should have raised ValueError')

    def test_raises_if_state_quantity_units_undefined(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
            ),
        }
        try:
            return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        except InvalidStateError:
            pass
        else:
            raise AssertionError('should have raised InvalidStateError')

    def test_raises_if_no_units_undefined(self):
        property_dictionary = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
            ),
        }
        try:
            return_value = get_numpy_arrays_with_properties(state, property_dictionary)
        except InvalidPropertyDictError:
            pass
        except InvalidStateError:
            pass
        else:
            raise AssertionError('should have raised InvalidPropertyDictError or InvalidStateError')

    def test_raises_if_dims_property_not_specified(self):
        property_dictionary = {
            'air_temperature': {
                'units': 'degK',
            },
        }
        state = {
            'air_temperature': DataArray(
                np.zeros([4], dtype=np.float64),
                dims=['z'],
                attrs={'units': 'degK'},
            ),
        }
        try:
            get_numpy_arrays_with_properties(state, property_dictionary)
        except InvalidPropertyDictError:
            pass
        else:
            raise AssertionError('should have raised ValueError')

    def test_collects_horizontal_dimensions(self):
        random = np.random.RandomState(0)
        T_array = random.randn(3, 2, 4)
        input_state = {
            'air_temperature': DataArray(
                T_array,
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['z', '*'],
                'units': 'degK',
            }
        }
        return_value = get_numpy_arrays_with_properties(input_state, input_properties)
        assert return_value['air_temperature'].shape == (4, 6)
        for i in range(3):
            for j in range(2):
                for k in range(4):
                    assert return_value['air_temperature'][k, j+2*i] == T_array[i, j, k]

    def test_raises_when_quantity_has_extra_dim(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2,4]),
                dims=['foo', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['z'],
                'units': 'degK',
            }
        }
        try:
            get_numpy_arrays_with_properties(input_state, input_properties)
        except InvalidStateError:
            pass
        else:
            raise AssertionError('should have raised InvalidStateError')

    def test_raises_when_quantity_has_extra_dim_and_unmatched_wildcard(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 4]),
                dims=['foo', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['y', 'z'],
                'units': 'degK',
            }
        }
        try:
            get_numpy_arrays_with_properties(input_state, input_properties)
        except InvalidStateError:
            pass
        else:
            raise AssertionError('should have raised InvalidStateError')

    def test_expands_named_dimension(self):
        random = np.random.RandomState(0)
        T_array = random.randn(3)
        input_state = {
            'air_pressure': DataArray(
                np.zeros([3, 4]),
                dims=['dim1', 'dim2'],
                attrs={'units': 'Pa'},
            ),
            'air_temperature': DataArray(
                T_array,
                dims=['dim1'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_pressure': {
                'dims': ['dim1', 'dim2'],
                'units': 'Pa',
            },
            'air_temperature': {
                'dims': ['dim1', 'dim2'],
                'units': 'degK',
            },
        }
        return_value = get_numpy_arrays_with_properties(input_state, input_properties)
        assert return_value['air_temperature'].shape == (3, 4)
        assert np.all(return_value['air_temperature'] == T_array[:, None])

    def test_expands_named_dimension_with_wildcard_present(self):
        random = np.random.RandomState(0)
        T_array = random.randn(3)
        input_state = {
            'air_pressure': DataArray(
                np.zeros([3, 4]),
                dims=['dim1', 'dim2'],
                attrs={'units': 'Pa'},
            ),
            'air_temperature': DataArray(
                T_array,
                dims=['dim1'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_pressure': {
                'dims': ['*', 'dim2'],
                'units': 'Pa',
            },
            'air_temperature': {
                'dims': ['*', 'dim2'],
                'units': 'degK',
            },
        }
        return_value = get_numpy_arrays_with_properties(input_state, input_properties)
        assert return_value['air_temperature'].shape == (3, 4)
        assert np.all(return_value['air_temperature'] == T_array[:, None])


class RestoreDataArraysWithPropertiesTests(unittest.TestCase):

    def test_restores_with_dims(self):
        raw_arrays = {
            'output1': np.ones([10]),
        }
        output_properties =  {
            'output1': {
                'dims': ['dim1'],
                'units': 'm'
            }
        }
        output = restore_data_arrays_with_properties(
            raw_arrays, output_properties, {}, {})
        assert len(output) == 1
        assert 'output1' in output.keys()
        assert isinstance(output['output1'], DataArray)
        assert len(output['output1'].dims) == 1
        assert 'dim1' in output['output1'].dims
        assert 'units' in output['output1'].attrs.keys()
        assert output['output1'].attrs['units'] == 'm'

    def test_returns_simple_value(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            }
        }
        raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
        raw_arrays = {key + '_tendency': value for key, value in raw_arrays.items()}
        output_properties = {
            'air_temperature_tendency': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK/s',
            }
        }
        return_value = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature_tendency'], DataArray)
        assert return_value['air_temperature_tendency'].attrs['units'] is 'degK/s'
        assert byte_bounds(
            return_value['air_temperature_tendency'].values) == byte_bounds(
            input_state['air_temperature'].values)
        assert (arrays_share_same_memory_space(
            return_value['air_temperature_tendency'].values, 
            input_state['air_temperature'].values))
        assert return_value['air_temperature_tendency'].shape == (2, 2, 4)


    def test_assumes_dims_like_own_name(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            }
        }
        raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
        output_properties = {
            'air_temperature': {
                'units': 'degK/s',
            }
        }
        return_value = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature'], DataArray)
        assert return_value['air_temperature'].attrs['units'] is 'degK/s'
        assert byte_bounds(
            return_value['air_temperature'].values) == byte_bounds(
            input_state['air_temperature'].values)
        assert (arrays_share_same_memory_space(
            return_value['air_temperature'].values, 
            input_state['air_temperature'].values))
        assert return_value['air_temperature'].shape == (2, 2, 4)

    def test_restores_collected_horizontal_dimensions(self):
        random = np.random.RandomState(0)
        T_array = random.randn(3, 2, 4)
        input_state = {
            'air_temperature': DataArray(
                T_array,
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['z', '*'],
                'units': 'degK',
            }
        }
        raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
        raw_arrays = {key + '_tendency': value for key, value in raw_arrays.items()}
        output_properties = {
            'air_temperature_tendency': {
                'dims': ['z', '*'],
                'units': 'degK/s',
            }
        }
        return_value = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert isinstance(return_value, dict)
        assert len(return_value.keys()) == 1
        assert isinstance(return_value['air_temperature_tendency'], DataArray)
        assert return_value['air_temperature_tendency'].attrs['units'] is 'degK/s'
        assert byte_bounds(
            return_value['air_temperature_tendency'].values) == byte_bounds(
            input_state['air_temperature'].values)
        assert (arrays_share_same_memory_space(
            input_state['air_temperature'].values,
            return_value['air_temperature_tendency'].values))
        assert return_value['air_temperature_tendency'].dims == ('z', 'x', 'y')
        assert return_value['air_temperature_tendency'].shape == (4, 3, 2)
        for i in range(4):
            assert np.all(return_value['air_temperature_tendency'][i, :, :] == T_array[:, :, i])

    def test_restores_scalar_array(self):
        T_array = np.array(0.)
        input_properties = {
            'surface_temperature': {
                'units': 'degK',
                'dims': ['*'],
            },
        }
        input_state = {
            'surface_temperature': DataArray(
                T_array,
                dims=[],
                attrs={'units': 'degK'},
            ),
        }
        raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
        output_properties = {
            'surface_temperature': {
                'units': 'degK',
            }
        }
        return_value = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert len(return_value.keys()) == 1
        assert 'surface_temperature' in return_value.keys()
        assert len(return_value['surface_temperature'].values.shape) == 0
        assert return_value['surface_temperature'].attrs['units'] == 'degK'

    def test_raises_on_invalid_shape(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            }
        }
        raw_arrays = {
            'foo': np.zeros([2, 4])
        }
        output_properties = {
            'foo': {
                'dims': ['x', 'y', 'z'],
                'units': 'm',
            }
        }
        try:
            restore_data_arrays_with_properties(
                raw_arrays, output_properties, input_state, input_properties
            )
        except InvalidPropertyDictError:
            pass
        else:
            raise AssertionError('should have raised InvalidPropertyDictError')

    def test_raises_on_raw_array_missing(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            }
        }
        raw_arrays = {
            'foo': np.zeros([2, 2, 4])
        }
        output_properties = {
            'foo': {
                'dims': ['x', 'y', 'z'],
                'units': 'm',
            },
            'bar': {
                'dims': ['x', 'y', 'z'],
                'units': 'm',
            },
        }
        try:
            restore_data_arrays_with_properties(
                raw_arrays, output_properties, input_state, input_properties
            )
        except KeyError:
            pass
        else:
            raise AssertionError('should have raised KeyError')

    def test_restores_aliased_name(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            )
        }
        input_properties = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            }
        }
        raw_arrays = {
            'p': np.zeros([2, 2, 4])
        }
        output_properties = {
            'air_pressure': {
                'dims': ['x', 'y', 'z'],
                'units': 'm',
                'alias': 'p',
            },
        }
        data_arrays = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert len(data_arrays.keys()) == 1
        assert 'air_pressure' in data_arrays.keys()
        assert np.all(data_arrays['air_pressure'].values == raw_arrays['p'])
        assert byte_bounds(data_arrays['air_pressure'].values) == byte_bounds(raw_arrays['p'])

    def test_restores_using_alias_from_input(self):
        input_state = {
            'air_temperature': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            ),
            'air_pressure': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            ),
        }
        input_properties = {
            'air_temperature': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
            },
            'air_pressure': {
                'dims': ['x', 'y', 'z'],
                'units': 'degK',
                'alias': 'p'
            },
        }
        raw_arrays = {
            'p': np.zeros([2, 2, 4])
        }
        output_properties = {
            'air_pressure': {
                'dims': ['x', 'y', 'z'],
                'units': 'm',
            },
        }
        data_arrays = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert len(data_arrays.keys()) == 1
        assert 'air_pressure' in data_arrays.keys()
        assert np.all(data_arrays['air_pressure'].values == raw_arrays['p'])
        assert byte_bounds(
            data_arrays['air_pressure'].values) == byte_bounds(
            raw_arrays['p'])

    def test_restores_new_dims(self):
        input_state = {}
        input_properties = {}
        raw_arrays = {
            'air_pressure': np.zeros([2, 2, 4])
        }
        output_properties = {
            'air_pressure': {
                'dims': ['x', 'y', 'z'],
                'units': 'm',
            },
        }
        data_arrays = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert len(data_arrays.keys()) == 1
        assert 'air_pressure' in data_arrays.keys()
        assert np.all(data_arrays['air_pressure'].values == raw_arrays['air_pressure'])
        assert byte_bounds(
            data_arrays['air_pressure'].values) == byte_bounds(
            raw_arrays['air_pressure'])

    def test_restores_new_dims_with_wildcard(self):
        input_state = {
            'air_pressure': DataArray(
                np.zeros([2, 2, 4]),
                dims=['x', 'y', 'z'],
                attrs={'units': 'degK'},
            ),
        }
        input_properties = {
            'air_pressure': {
                'dims': ['*'],
                'units': 'degK',
                'alias': 'p'
            },
        }
        raw_arrays = {
            'q': np.zeros([16, 2])
        }
        output_properties = {
            'q': {
                'dims': ['*', 'new_dim'],
                'units': 'm',
            },
        }
        data_arrays = restore_data_arrays_with_properties(
            raw_arrays, output_properties, input_state, input_properties
        )
        assert len(data_arrays.keys()) == 1
        assert 'q' in data_arrays.keys()
        assert np.all(data_arrays['q'].values.flatten() == raw_arrays['q'].flatten())
        assert byte_bounds(
            data_arrays['q'].values) == byte_bounds(
            raw_arrays['q'])
        assert data_arrays['q'].dims == ('x', 'y', 'z', 'new_dim')
        assert data_arrays['q'].shape == (2, 2, 4, 2)


if __name__ == '__main__':
    pytest.main([__file__])
