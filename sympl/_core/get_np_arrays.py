import numpy as np
from .exceptions import InvalidStateError
from .wildcard import get_wildcard_matches_and_dim_lengths, flatten_wildcard_dims


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


def ensure_quantity_has_units(quantity, quantity_name):
    if 'units' not in quantity.attrs:
        raise InvalidStateError(
            'quantity {} is missing units attribute'.format(quantity_name))
