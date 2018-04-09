import unittest
from sympl import initialize_numpy_arrays_with_properties, DataArray
import numpy as np


class InitializeNumpyArraysWithPropertiesTests(unittest.TestCase):

    def test_empty(self):
        output_properties = {
        }
        input_properties = {
        }
        input_state = {
        }
        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert result == {}

    def test_single_output_single_dim(self):
        output_properties = {
            'output1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['dim1'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([10]),
                dims=['dim1'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (10,)
        assert np.all(result['output1'] == np.zeros([10]))

    def test_single_output_two_dims(self):
        output_properties = {
            'output1': {
                'dims': ['dim1', 'dim2'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([3, 7]),
                dims=['dim1', 'dim2'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (3, 7)
        assert np.all(result['output1'] == np.zeros([3, 7]))

    def test_single_output_two_dims_opposite_order(self):
        output_properties = {
            'output1': {
                'dims': ['dim2', 'dim1'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([3, 7]),
                dims=['dim1', 'dim2'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (7, 3)
        assert np.all(result['output1'] == np.zeros([7, 3]))

    def test_two_outputs(self):
        output_properties = {
            'output1': {
                'dims': ['dim2', 'dim1'],
                'units': 'm',
            },
            'output2': {
                'dims': ['dim1', 'dim2'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([3, 7]),
                dims=['dim1', 'dim2'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 2
        assert 'output1' in result.keys()
        assert result['output1'].shape == (7, 3)
        assert np.all(result['output1'] == np.zeros([7, 3]))
        assert 'output2' in result.keys()
        assert result['output2'].shape == (3, 7)
        assert np.all(result['output2'] == np.zeros([3, 7]))

    def test_two_inputs(self):
        output_properties = {
            'output1': {
                'dims': ['dim2', 'dim1'],
                'units': 'm',
            },
        }
        input_properties = {
            'input1': {
                'dims': ['dim1', 'dim2'],
                'units': 's^-1',
            },
            'input2': {
                'dims': ['dim2', 'dim1'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([3, 7]),
                dims=['dim1', 'dim2'],
                attrs={'units': 's^-1'}
            ),
            'input2': DataArray(
                np.zeros([7, 3]),
                dims=['dim2', 'dim1'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (7, 3)
        assert np.all(result['output1'] == np.zeros([7, 3]))

    def test_single_dim_wildcard(self):
        output_properties = {
            'output1': {
                'dims': ['*'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([10]),
                dims=['dim1'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (10,)
        assert np.all(result['output1'] == np.zeros([10]))

    def test_two_dims_in_wildcard(self):
        output_properties = {
            'output1': {
                'dims': ['*'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['*'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([4, 3]),
                dims=['dim1', 'dim2'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (12,)
        assert np.all(result['output1'] == np.zeros([12]))

    def test_two_dims_in_wildcard_with_basic_dim(self):
        output_properties = {
            'output1': {
                'dims': ['*', 'dim3'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['*', 'dim3'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([4, 3, 2]),
                dims=['dim1', 'dim2', 'dim3'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (12, 2)
        assert np.all(result['output1'] == np.zeros([12, 2]))

    def test_two_dims_in_wildcard_with_basic_dim_in_center(self):
        output_properties = {
            'output1': {
                'dims': ['*', 'dim2'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['*', 'dim2'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([4, 3, 2]),
                dims=['dim1', 'dim2', 'dim3'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (8, 3)
        assert np.all(result['output1'] == np.zeros([8, 3]))

    def test_two_dims_in_wildcard_with_basic_dim_and_extra_basic_dim_in_input(self):
        output_properties = {
            'output1': {
                'dims': ['*', 'dim3'],
                'units': 'm',
            }
        }
        input_properties = {
            'input1': {
                'dims': ['*', 'dim3', 'dim4'],
                'units': 's^-1',
            }
        }
        input_state = {
            'input1': DataArray(
                np.zeros([4, 3, 2, 5]),
                dims=['dim1', 'dim2', 'dim3', 'dim4'],
                attrs={'units': 's^-1'}
            )
        }

        result = initialize_numpy_arrays_with_properties(
            output_properties, input_state, input_properties)
        assert len(result.keys()) == 1
        assert 'output1' in result.keys()
        assert result['output1'].shape == (12, 2)
        assert np.all(result['output1'] == np.zeros([12, 2]))
