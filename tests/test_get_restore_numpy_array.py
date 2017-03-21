import pytest
from sympl import (
    DataArray, set_dimension_names, get_numpy_array,
    restore_dimensions)
import numpy as np


def test_get_numpy_array_3d_no_change():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['x', 'y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['x', 'y', 'z'])
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert np.all(numpy_array == array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_3d_reverse():
    array = DataArray(
        np.random.randn(2, 3, 4),
        dims=['x', 'y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['z', 'y', 'x'])
    assert numpy_array.shape == (4, 3, 2)
    assert np.all(np.transpose(numpy_array, (2, 1, 0)) == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_2d_reverse():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['z', 'y'])
    assert numpy_array.shape == (3, 2)
    assert np.all(np.transpose(numpy_array, (1, 0)) == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_1d():
    array = DataArray(
        np.random.randn(2),
        dims=['y'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['y'])
    assert numpy_array.shape == (2,)
    assert np.all(numpy_array == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_creates_new_dim():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['x', 'y'])
    assert numpy_array.shape == (2, 1)
    assert np.all(numpy_array[:, 0] == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_creates_new_dim_in_front():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['y', 'x'])
    assert numpy_array.shape == (1, 2)
    assert np.all(numpy_array[0, :] == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_retrieves_explicit_dimensions():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['alpha', 'zeta'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['zeta', 'alpha'])
    assert numpy_array.shape == (3, 2)
    assert np.all(np.transpose(numpy_array, (1, 0)) == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


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
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


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


def test_get_numpy_array_multiple_dims_on_same_direction():
    try:
        set_dimension_names(x=['lon'])
        array = DataArray(
            np.random.randn(2, 3),
            dims=['x', 'lon'],
            attrs={'units': ''},
        )
        try:
            numpy_array = get_numpy_array(array, ['x', 'y'])
        except ValueError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError('Expected ValueError but no error was raised')
    finally:
        set_dimension_names(x=[], y=[], z=[])


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
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_asterisk_creates_new_dim_reversed():
    array = DataArray(
        np.random.randn(2),
        dims=['x'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['*', 'x'])
    assert numpy_array.shape == (1, 2)
    assert np.all(numpy_array[0, :] == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


def test_get_numpy_array_asterisk_flattens():
    array = DataArray(
        np.random.randn(2, 3),
        dims=['y', 'z'],
        attrs={'units': ''},
    )
    numpy_array = get_numpy_array(array, ['*'])
    assert numpy_array.shape == (6,)
    assert np.all(numpy_array.reshape((2, 3)) == array.values)
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


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
    assert np.byte_bounds(numpy_array) == np.byte_bounds(array.values)
    assert numpy_array.base is array.values


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
    assert np.byte_bounds(restored_array.values) == np.byte_bounds(
        array.values)
    assert restored_array.values.base is array.values


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
    assert np.byte_bounds(restored_array.values) == np.byte_bounds(
        array.values)
    assert restored_array.values.base is array.values


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
    assert np.byte_bounds(restored_array.values) == np.byte_bounds(
        array.values)
    assert restored_array.values.base is array.values


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
    assert np.byte_bounds(restored_array.values) == np.byte_bounds(
        array.values)
    assert restored_array.values.base is array.values


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
    assert np.byte_bounds(restored_array.values) == np.byte_bounds(
        array.values)
    assert restored_array.values.base is array.values

if __name__ == '__main__':
    pytest.main([__file__])
