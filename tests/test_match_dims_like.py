import pytest
from sympl import (
    DataArray, get_numpy_arrays_with_properties, InvalidStateError)
import numpy as np


try:
    from numpy.lib.array_utils import byte_bounds
except ImportError:
    from numpy import byte_bounds


def test_match_dims_like_hardcoded_dimensions_matching_lengths():
    input_state = {
        'air_temperature': DataArray(
            np.zeros([2, 3, 4]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'degK'},
        ),
        'air_pressure': DataArray(
            np.zeros([2, 3, 4]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'Pa'},
        ),
    }
    input_properties = {
        'air_temperature': {
            'dims': ['alpha', 'beta', 'gamma'],
            'units': 'degK',
            'match_dims_like': 'air_pressure',
        },
        'air_pressure': {
            'dims': ['alpha', 'beta', 'gamma'],
            'units': 'Pa',
        },
    }
    raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)


def test_match_dims_like_partly_hardcoded_dimensions_matching_lengths():
    input_state = {
        'air_temperature': DataArray(
            np.zeros([2, 3, 4]),
            dims=['lat', 'lon', 'mid_levels'],
            attrs={'units': 'degK'},
        ),
        'air_pressure': DataArray(
            np.zeros([2, 3, 4]),
            dims=['lat', 'lon', 'interface_levels'],
            attrs={'units': 'Pa'},
        ),
    }
    input_properties = {
        'air_temperature': {
            'dims': ['*', 'mid_levels'],
            'units': 'degK',
            'match_dims_like': 'air_pressure',
        },
        'air_pressure': {
            'dims': ['*', 'interface_levels'],
            'units': 'Pa',
        },
    }
    raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
    assert byte_bounds(input_state['air_temperature'].values) == byte_bounds(raw_arrays['air_temperature'])
    assert byte_bounds(input_state['air_pressure'].values) == byte_bounds(raw_arrays['air_pressure'])


def test_match_dims_like_hardcoded_dimensions_non_matching_lengths():
    input_state = {
        'air_temperature': DataArray(
            np.zeros([2, 3, 4]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'degK'},
        ),
        'air_pressure': DataArray(
            np.zeros([4, 2, 3]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'Pa'},
        ),
    }
    input_properties = {
        'air_temperature': {
            'dims': ['alpha', 'beta', 'gamma'],
            'units': 'degK',
            'match_dims_like': 'air_pressure',
        },
        'air_pressure': {
            'dims': ['alpha', 'beta', 'gamma'],
            'units': 'Pa',
        },
    }
    try:
        raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
    except InvalidStateError:
        pass
    else:
        raise AssertionError('should have raised InvalidStateError')


def test_match_dims_like_wildcard_dimensions_matching_lengths():
    input_state = {
        'air_temperature': DataArray(
            np.zeros([2, 3, 4]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'degK'},
        ),
        'air_pressure': DataArray(
            np.zeros([2, 3, 4]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'Pa'},
        ),
    }
    input_properties = {
        'air_temperature': {
            'dims': ['*'],
            'units': 'degK',
            'match_dims_like': 'air_pressure',
        },
        'air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
        },
    }
    raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)


def test_match_dims_like_wildcard_dimensions_non_matching_lengths():
    input_state = {
        'air_temperature': DataArray(
            np.zeros([2, 3, 4]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'degK'},
        ),
        'air_pressure': DataArray(
            np.zeros([1, 2, 3]),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'Pa'},
        ),
    }
    input_properties = {
        'air_temperature': {
            'dims': ['*'],
            'units': 'degK',
            'match_dims_like': 'air_pressure',
        },
        'air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
        },
    }
    try:
        raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
    except InvalidStateError:
        pass
    else:
        raise AssertionError('should have raised InvalidStateError')


def test_match_dims_like_wildcard_dimensions_use_same_ordering():
    input_state = {
        'air_temperature': DataArray(
            np.random.randn(2, 3, 4),
            dims=['alpha', 'beta', 'gamma'],
            attrs={'units': 'degK'},
        ),
        'air_pressure': DataArray(
            np.zeros([4, 2, 3]),
            dims=['gamma', 'alpha', 'beta'],
            attrs={'units': 'Pa'},
        ),
    }
    for i in range(4):
        input_state['air_pressure'][i, :, :] = input_state['air_temperature'][:, :, i]
    input_properties = {
        'air_temperature': {
            'dims': ['*'],
            'units': 'degK',
            'match_dims_like': 'air_pressure',
        },
        'air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
        },
    }
    raw_arrays = get_numpy_arrays_with_properties(input_state, input_properties)
    assert np.all(raw_arrays['air_temperature'] == raw_arrays['air_pressure'])


if __name__ == '__main__':
    pytest.main([__file__])
