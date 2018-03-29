from .exceptions import InvalidStateError, InvalidPropertyDictError
import numpy as np
from .array import DataArray


def copy_untouched_quantities(old_state, new_state):
    for key in old_state.keys():
        if key not in new_state:
            new_state[key] = old_state[key]


def add(state_1, state_2):
    out_state = {}
    if 'time' in state_1.keys():
        out_state['time'] = state_1['time']
    for key in state_1.keys():
        if key != 'time':
            out_state[key] = state_1[key] + state_2[key]
            if hasattr(out_state[key], 'attrs'):
                out_state[key].attrs = state_1[key].attrs
    return out_state


def multiply(scalar, state):
    out_state = {}
    if 'time' in state.keys():
        out_state['time'] = state['time']
    for key in state.keys():
        if key != 'time':
            out_state[key] = scalar * state[key]
            if hasattr(out_state[key], 'attrs'):
                out_state[key].attrs = state[key].attrs
    return out_state


def get_wildcard_matches_and_dim_lengths(state, property_dictionary):
    wildcard_names = []
    dim_lengths = {}
    # Loop to get the set of names matching "*" (wildcard names)
    for quantity_name, properties in property_dictionary.items():
        ensure_properties_have_dims_and_units(properties, quantity_name)
        for dim_name, length in zip(state[quantity_name].dims, state[quantity_name].shape):
            if dim_name not in dim_lengths.keys():
                dim_lengths[dim_name] = length
            elif length != dim_lengths[dim_name]:
                raise InvalidStateError(
                    'Dimension {} conflicting lengths {} and {} in different '
                    'state quantities.'.format(dim_name, length, dim_lengths[dim_name]))
        new_wildcard_names = [
            dim for dim in state[quantity_name].dims if dim not in properties['dims']]
        if len(new_wildcard_names) > 0 and '*' not in properties['dims']:
            raise InvalidStateError(
                'Quantity {} has unexpected dimensions {}.'.format(
                    quantity_name, new_wildcard_names))
        wildcard_names.extend(
            [name for name in new_wildcard_names if name not in wildcard_names])
    wildcard_names = tuple(wildcard_names)
    return wildcard_names, dim_lengths


def get_numpy_arrays_with_properties(state, property_dictionary):
    out_dict = {}
    wildcard_names, dim_lengths = get_wildcard_matches_and_dim_lengths(
        state, property_dictionary)
    #  Now we actually retrieve output arrays since we know the precise out dims
    for name, properties in property_dictionary.items():
        ensure_quantity_has_units(state[name], name)
        try:
            quantity = state[name].to_units(properties['units'])
        except ValueError:
            raise InvalidStateError(
                'Could not convert quantity {} from units {} to units {}'.format(
                    name, state[name].attrs['units'], properties['units']
                )
            )
        out_dims = []
        out_dims.extend(properties['dims'])
        has_wildcard = '*' in out_dims
        if has_wildcard:
            i_wildcard = out_dims.index('*')
            out_dims[i_wildcard:i_wildcard+1] = wildcard_names
        out_array = get_numpy_array(
            quantity, out_dims=out_dims, dim_lengths=dim_lengths)
        if has_wildcard:
            out_array = flatten_wildcard_dims(
                out_array, i_wildcard, i_wildcard + len(wildcard_names))
        if 'alias' in properties.keys():
            out_name = properties['alias']
        else:
            out_name = name
        out_dict[out_name] = out_array
    return out_dict


def flatten_wildcard_dims(array, i_start, i_end):
    if i_end > len(array.shape):
        raise ValueError('i_end should be less than the number of axes in array')
    elif i_start < 0:
        raise ValueError('i_start should be greater than 0')
    elif i_start > i_end:
        raise ValueError('i_start should be less than or equal to i_end')
    elif i_start == i_end:
        # We need to insert a singleton dimension at i_start
        target_shape = []
        target_shape.extend(array.shape)
        target_shape.insert(i_start, 1)
    else:
        target_shape = []
        wildcard_length = 1
        for i, length in enumerate(array.shape):
            if i_start <= i < i_end:
                wildcard_length *= length
            else:
                target_shape.append(length)
            if i == i_end - 1:
                target_shape.append(wildcard_length)
    return array.reshape(target_shape)


def get_numpy_array(data_array, out_dims, dim_lengths):
    """
    Gets a numpy array from the data_array with the desired out_dims, and a
    dict of dim_lengths that will give the length of any missing dims in the
    data_array.
    """
    if len(data_array.values.shape) == 0 and len(out_dims) == 0:
        return data_array.values  # special case, 0-dimensional scalar array
    else:
        missing_dims = [dim for dim in out_dims if dim not in data_array.dims]
        for dim in missing_dims:
            data_array = data_array.expand_dims(dim)
        numpy_array = data_array.transpose(*out_dims).values
        if len(missing_dims) == 0:
            out_array = numpy_array
        else:  # expand out missing dims which are currently length 1.
            out_shape = [dim_lengths.get(name, 1) for name in out_dims]
            if out_shape == list(numpy_array.shape):
                out_array = numpy_array
            else:
                out_array = np.empty(out_shape, dtype=numpy_array.dtype)
                out_array[:] = numpy_array
        return out_array


def restore_data_arrays_with_properties(
        raw_arrays, output_properties, input_state, input_properties,
        ignore_names=None):
    if ignore_names is None:
        ignore_names = []
    wildcard_names, dim_lengths = get_wildcard_matches_and_dim_lengths(
        input_state, input_properties)
    for name, value in raw_arrays.items():
        if not isinstance(value, np.ndarray):
            raw_arrays[name] = np.asarray(value)
    out_dims_property = {}
    for name, properties in output_properties.items():
        if name in ignore_names:
            continue
        elif 'dims' in properties.keys():
            out_dims_property[name] = properties['dims']
        elif name not in input_properties.keys():
            raise InvalidPropertyDictError(
                'Output dims must be specified for {} in properties'.format(name))
        elif 'dims' not in input_properties[name].keys():
            raise InvalidPropertyDictError(
                'Input dims must be specified for {} in properties'.format(name))
        else:
            out_dims_property[name] = input_properties[name]['dims']
    out_dict = {}
    for name, dims in out_dims_property.items():
        if name in ignore_names:
            continue
        if 'alias' in output_properties[name].keys():
            raw_name = output_properties[name]['alias']
        elif name in input_properties.keys() and 'alias' in input_properties[name].keys():
            raw_name = input_properties[name]['alias']
        else:
            raw_name = name
        if '*' in dims:
            i_wildcard = dims.index('*')
            target_shape = []
            out_dims = []
            for i, length in enumerate(raw_arrays[raw_name].shape):
                if i == i_wildcard:
                    target_shape.extend([dim_lengths[n] for n in wildcard_names])
                    out_dims.extend(wildcard_names)
                else:
                    target_shape.append(length)
                    out_dims.append(dims[i])
            out_array = np.reshape(raw_arrays[raw_name], target_shape)
        else:
            if len(dims) != len(raw_arrays[raw_name].shape):
                raise InvalidPropertyDictError(
                    'Returned array for {} has shape {} '
                    'which is incompatible with dims {} in properties'.format(
                        name, raw_arrays[raw_name].shape, dims))
            for dim, length in zip(dims, raw_arrays[raw_name].shape):
                if dim in dim_lengths.keys() and dim_lengths[dim] != length:
                    raise InvalidPropertyDictError(
                        'Dimension {} of quantity {} has length {}, but '
                        'another quantity has length {}'.format(
                            dim, name, length, dim_lengths[dim])
                    )
            out_dims = dims
            out_array = raw_arrays[raw_name]
        out_dict[name] = DataArray(
            out_array,
            dims=out_dims,
            attrs={'units': output_properties[name]['units']}
        )
    return out_dict


def ensure_properties_have_dims_and_units(properties, quantity_name):
    if 'dims' not in properties:
        raise InvalidPropertyDictError(
            'dims not specified for quantity {}'.format(quantity_name))
    if 'units' not in properties:
        raise InvalidPropertyDictError(
            'units not specified for quantity {}'.format(quantity_name))


def ensure_quantity_has_units(quantity, quantity_name):
    if 'units' not in quantity.attrs:
        raise InvalidStateError(
            'quantity {} is missing units attribute'.format(quantity_name))
