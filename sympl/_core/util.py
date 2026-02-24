from datetime import datetime

import numpy as np

from .dataarray import DataArray
from .exceptions import (
    SharedKeyError, InvalidStateError)

try:
    from numba import jit
except ImportError:
    # define a function with the same call signature as jit that does nothing
    def jit(signature_or_function=None, **kwargs):
        if signature_or_function is None:
            return lambda x: x
        else:
            return signature_or_function

# internal exceptions used only within this module


class NoMatchForDirectionError(Exception):
    pass


class DimensionNotInOutDimsError(ValueError):
    pass


class ShapeMismatchError(Exception):
    pass


def get_numpy_array(
        data_array, out_dims, return_wildcard_matches=False,
        require_wildcard_matches=None):
    """
    Retrieve a numpy array with the desired dimensions and dimension order
    from the given DataArray, using transpose and creating length 1 dimensions
    as necessary.

    Args
    ----
    data_array : DataArray
        The object from which to retrieve data.
    out_dims : list of str
        The desired dimensions of the output and their order.
        Length 1 dimensions will be created if the dimension is 'x', 'y', 'z',
        or '*' and does not exist in data_array. 'x', 'y', and 'z' indicate any axes
        registered to those directions with
        :py:function:`~sympl.set_direction_names`. '*' indicates an axis
        which is the flattened collection of all dimensions not explicitly
        listed in out_dims, including any dimensions with unknown direction.
    return_wildcard_matches : bool, optional
        If True, will additionally return a dictionary whose keys are direction
        wildcards (currently only '*') and values are lists of matched
        dimensions in the order they appear.
    require_wildcard_matches : dict, optional
        A dictionary mapping wildcards to matches. If the wildcard is used in
        out_dims, ensures that it matches the quantities present in this
        dictionary, in the same order.

    Returns
    -------
    numpy_array : ndarray
        The desired array, with dimensions in the
        correct order and length 1 dimensions created as needed.

    Raises
    ------
    ValueError
        If out_dims has values that are incompatible with the dimensions
        in data_array, or data_array's dimensions are invalid in some way.

    """
    # This function was written when we had directional wildcards, and could
    # be re-written to be simpler now that we do not.
    if (len(data_array.values.shape) == 0) and (len(out_dims) == 0):
        direction_to_names = {}  # required in case we need wildcard_matches
        return_array = data_array.values  # special case, 0-dimensional scalar array
    else:
        current_dim_names = {}
        for dim in out_dims:
            if dim != '*':
                current_dim_names[dim] = [dim]
        direction_to_names = get_input_array_dim_names(
            data_array, out_dims, current_dim_names)
        if require_wildcard_matches is not None:
            for direction in out_dims:
                if (direction in require_wildcard_matches and
                        same_list(direction_to_names[direction],
                                  require_wildcard_matches[direction])):
                    direction_to_names[direction] = require_wildcard_matches[
                        direction]
                else:
                    # we could raise an exception here, because this is
                    # inconsistent, but that exception is already raised
                    # elsewhere when ensure_dims_like_are_satisfied is called
                    pass
        target_dimension_order = get_target_dimension_order(
            out_dims, direction_to_names)
        for dim in data_array.dims:
            if dim not in target_dimension_order:
                raise DimensionNotInOutDimsError(dim)
        slices_or_none = get_slices_and_placeholder_nones(
            data_array, out_dims, direction_to_names)
        final_shape = get_final_shape(data_array, out_dims, direction_to_names)
        return_array = np.reshape(data_array.transpose(
            *target_dimension_order).values[tuple(slices_or_none)], final_shape)
    if return_wildcard_matches:
        wildcard_matches = {
            key: value for key, value in direction_to_names.items()
            if key == '*'}
        return return_array, wildcard_matches
    else:
        return return_array


def restore_dimensions(array, from_dims, result_like, result_attrs=None):
    """
    Restores a numpy array to a DataArray with similar dimensions to a reference
    Data Array. This is meant to be the reverse of get_numpy_array.

    Parameters
    ----------
    array : ndarray
        The numpy array from which to create a DataArray
    from_dims : list of str
        The directions describing the numpy array. If being used to reverse
        a call to get_numpy_array, this should be the same as the out_dims
        argument used in the call to get_numpy_array.
        'x', 'y', and 'z' indicate any axes
        registered to those directions with
        :py:function:`~sympl.set_direction_names`. '*' indicates an axis
        which is the flattened collection of all dimensions not explicitly
        listed in out_dims, including any dimensions with unknown direction.
    result_like : DataArray
        A reference array with the desired output dimensions of the DataArray.
        If being used to reverse a call to get_numpy_array, this should be
        the same as the data_array argument used in the call to get_numpy_array.
    result_attrs : dict, optional
        A dictionary with the desired attributes of the output DataArray. If
        not given, no attributes will be set.

    Returns
    -------
    data_array : DataArray
        The output DataArray with the same dimensions as the reference
        DataArray.

    See Also
    --------
    :py:function:~sympl.get_numpy_array: : Retrieves a numpy array with desired
        dimensions from a given DataArray.
    """
    current_dim_names = {}
    for dim in from_dims:
        if dim != '*':
            current_dim_names[dim] = [dim]
    direction_to_names = get_input_array_dim_names(
        result_like, from_dims, current_dim_names)
    original_shape = []
    original_dims = []
    original_coords = []
    for direction in from_dims:
        if direction in direction_to_names.keys():
            for name in direction_to_names[direction]:
                original_shape.append(len(result_like.coords[name]))
                original_dims.append(name)
                original_coords.append(result_like.coords[name])
    if np.prod(array.shape) != np.prod(original_shape):
        raise ShapeMismatchError
    data_array = DataArray(
        np.reshape(array, original_shape),
        dims=original_dims,
        coords=original_coords).transpose(
            *list(result_like.dims))
    if result_attrs is not None:
        data_array.attrs = result_attrs
    return data_array


def datetime64_to_datetime(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)


def same_list(list1, list2):
    """Returns a boolean indicating whether the items in list1 are the same
    items present in list2 (ignoring order)."""
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


def update_dict_by_adding_another(dict1, dict2):
    """
    Takes two dictionaries. Add values in dict2 to the values in dict1, if
    present. If not present, create a new value in dict1 equal to the value in
    dict2. Addition is done in-place if the values are
    array-like, to avoid data copying. Units are handled if the values are
    DataArrays with a 'units' attribute.
    """
    for key in dict2.keys():
        if key not in dict1:
            if hasattr(dict2[key], 'copy'):
                dict1[key] = dict2[key].copy()
            else:
                dict1[key] = dict2[key]
        else:
            if (isinstance(dict1[key], DataArray) and isinstance(dict2[key], DataArray)):
                if 'units' not in dict1[key].attrs or 'units' not in dict2[key].attrs:
                    raise InvalidStateError(
                        'DataArray objects must have units property defined')
                try:
                    dict1[key] += dict2[key].to_units(dict1[key].attrs['units'])
                except ValueError:  # dict1[key] is missing a dimension present in dict2[key]
                    dict1[key] = dict1[key] + dict2[key].to_units(dict1[key].attrs['units'])
            else:
                dict1[key] += dict2[key]  # += is in-place addition operator
    return  # not returning anything emphasizes that this is in-place


def ensure_no_shared_keys(dict1, dict2):
    """
    Raises SharedKeyError if there exists a key present in both
    dictionaries.
    """
    shared_keys = set(dict1.keys()).intersection(dict2.keys())
    if len(shared_keys) > 0:
        raise SharedKeyError(
            'unexpected shared keys: {}'.format(shared_keys))


def get_input_array_dim_names(data_array, out_dims, dim_names):
    """
    Parameters
    ----------
    data_array : DataArray
    out_dims : iterable
        directions in dim_names that should be included in the output,
        in the order they should be included
    dim_names : dict
        a mapping from directions to dimension names that fall under that
        direction wildcard.

    Returns
    -------
    input_array_dim_names : dict
        A mapping from directions included in out_dims to the directions
        present in data_array that correspond to those directions
    """
    input_array_dim_names = {}
    for direction in out_dims:
        if direction != '*':
            matching_dims = set(
                data_array.dims).intersection(dim_names[direction])
            # must ensure matching dims are in right order
            input_array_dim_names[direction] = []
            for dim in data_array.dims:
                if dim in matching_dims:
                    input_array_dim_names[direction].append(dim)
            if (direction not in ('x', 'y', 'z', '*') and
                    len(input_array_dim_names[direction]) == 0):
                raise NoMatchForDirectionError(direction)
    if '*' in out_dims:
        matching_dims = set(
            data_array.dims).difference(set.union(set([]), *input_array_dim_names.values()))
        input_array_dim_names['*'] = []
        for dim in data_array.dims:
            if dim in matching_dims:
                input_array_dim_names['*'].append(dim)
    return input_array_dim_names


def get_target_dimension_order(out_dims, direction_to_names):
    """
    Takes in an iterable of directions ('x', 'y', 'z', or '*') and a dictionary
    mapping those directions to a list of names corresponding to those
    directions. Returns a list of names in the same order as in out_dims,
    preserving the order within direction_to_names for each direction.
    """
    target_dimension_order = []
    for direction in out_dims:
        target_dimension_order.extend(direction_to_names[direction])
    return target_dimension_order


def get_slices_and_placeholder_nones(data_array, out_dims, direction_to_names):
    """
    Takes in a DataArray, a desired ordering of output directions, and
    a dictionary mapping those directions to a list of names corresponding to
    those directions. Returns a list with the same ordering as out_dims that
    contains slices for out_dims that have corresponding names (as many slices
    as names, and spanning the entire dimension named), and None for out_dims
    without corresponding names.

    This returned list can be used to create length-1 axes for the dimensions
    that currently have no corresponding names in data_array.
    """
    slices_or_none = []
    for direction in out_dims:
        if len(direction_to_names[direction]) == 0:
            slices_or_none.append(None)
        elif (direction != '*') and (len(direction_to_names[direction]) > 1):
            raise ValueError(
                'DataArray has multiple dimensions for a single direction')
        else:
            for name in direction_to_names[direction]:
                slices_or_none.append(slice(0, len(data_array.coords[name])))
    return tuple(slices_or_none)


def get_final_shape(data_array, out_dims, direction_to_names):
    """
    Determine the final shape that data_array must be reshaped to in order to
    have one axis for each of the out_dims (for instance, combining all
    axes collected by the '*' direction).
    """
    final_shape = []
    for direction in out_dims:
        if len(direction_to_names[direction]) == 0:
            final_shape.append(1)
        else:
            # determine shape once dimensions for direction (usually '*') are combined
            final_shape.append(
                np.prod([len(data_array.coords[name])
                         for name in direction_to_names[direction]]))
    return tuple(final_shape)


def get_component_aliases(*args):
    """
    Returns aliases for variables in the properties of Components (TendencyComponent,
    DiagnosticComponent, Stepper, and ImplicitTendencyComponent objects).

    If multiple aliases are present for the same variable, the following
    properties have priority in descending order: input, output, diagnostic,
    tendency. If multiple components give different aliases at the same priority
    level, one is chosen arbitrarily.

    Args
    ----
    *args : Component
        Components from which to fetch variable aliases from the input_properties,
        output_properties, diagnostic_properties, and tendency_properties dictionaries

    Returns
    -------
    aliases : dict
        A dictionary mapping quantity names to aliases
    """
    return_dict = {}
    for property_type in (
            'tendency_properties', 'diagnostic_properties', 'output_properties',
            'input_properties'):
        for component in args:
            if hasattr(component, property_type):
                component_properties = getattr(component, property_type)
                for name, properties in component_properties.items():
                    if 'alias' in properties.keys():
                        return_dict[name] = properties['alias']
    return return_dict
