from .exceptions import SharedKeyException
from .constants import default_constants
from .array import DataArray
from six import string_types
import numpy as np
from datetime import datetime
try:
    from numba import jit
except ImportError:
    # define a function with the same call signature as jit that does nothing
    def jit(signature_or_function=None, **kwargs):
        if signature_or_function is None:
            return lambda x: x
        else:
            return signature_or_function

dim_names = {'x': ['x'], 'y': ['y'], 'z': ['z']}


def get_numpy_arrays_with_properties(property_dictionary, state):
    out_dict = {}
    for quantity_name, properties in property_dictionary.items():
        out_dict[quantity_name] = get_numpy_array(
            state[quantity_name].to_units(properties['units'], properties['dims']))
    return out_dict


def restore_data_arrays_with_properties(
        raw_arrays, property_dict, attr_dict, results_like_dict):
    out_dict = {}
    for quantity_name, array in raw_arrays.items():
        out_dict[quantity_name] = restore_dimensions(
            array,
            from_dims=property_dict[quantity_name]['dims'],
            result_like=results_like_dict[quantity_name],
            result_attrs=attr_dict[quantity_name])
    return out_dict


def datetime64_to_datetime(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)


def same_list(list1, list2):
    """Returns a boolean indicating whether the items in list1 are the same
    items present in list2 (ignoring order)."""
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


def set_dimension_names(x=None, y=None, z=None):
    for key, value in [('x', x), ('y', y), ('z', z)]:
        if isinstance(value, string_types):
            dim_names[key] = [key, value]
        elif value is not None:
            dim_names[key] = [key] + list(value)


def combine_dimensions(arrays, out_dims):
    """
    Returns a tuple of dimension names corresponding to
    dimension names from the DataArray objects given by *args when present.
    The names returned correspond to the directions in out_dims.

    Args
    ----
    arrays : iterable of DataArray
        Objects from which to deduce dimension names.
    out_dims : {'x', 'y', 'z'}
        The desired output directions. Should contain only 'x', 'y', or 'z'.
        For example, ('y', 'x') is valid.

    Raises
    ------
    ValueError
        If there are multiple names for a single direction, or if
        an array has a dimension along a direction not present in out_dims.

    Returns
    -------
    dimensions : list of str
        The deduced dimension names, in the order given by out_dims.
    """
    _ensure_no_invalid_directions(out_dims)
    out_names = [None for _ in range(len(out_dims))]
    all_names = set()
    for value in arrays:
        all_names.update(value.dims)
    for direction, dir_names in dim_names.items():
        if direction in out_dims:
            names = all_names.intersection(dir_names)
            if len(names) > 1:
                raise ValueError(
                    'Multiple dimensions along {} direction'.format(direction))
            elif len(names) == 1:
                out_names[out_dims.index(direction)] = names.pop()
            else:
                out_names[out_dims.index(direction)] = direction
        elif len(all_names.intersection(dir_names)) > 0:
            raise ValueError(
                'Arrays have dimensions along {} direction, which is '
                'not included in output'.format(direction))
    return out_names


def _ensure_no_invalid_directions(out_dims):
    invalid_dims = set(out_dims).difference(['x', 'y', 'z'])
    if len(invalid_dims) != 0:
        raise ValueError(
            'Invalid direction(s) in out_dims: {}'.format(invalid_dims))


class UpdateFrequencyWrapper(object):
    """
    Wraps a prognostic object so that when it is called, it only computes new
    output if sufficient time has passed, and otherwise returns its last
    computed output. The Delayed object requires that the 'time' attribute is
    set in the state, in addition to any requirements of the Prognostic

    Example
    -------
    This how the wrapper should be used on a fictional Prognostic class
    called MyPrognostic.

    >>> from datetime import timedelta
    >>> prognostic = Delayed(MyPrognostic(), timedelta(hours=1))
    """

    def __init__(self, prognostic, update_timedelta):
        """
        Initialize the Delayed object.

        Args
        ----
        prognostic : Prognostic
            The object to be wrapped
        update_timedelta : timedelta
            The amount that state['time'] must differ from when output
            was cached before new output is computed.
        """
        self._prognostic = prognostic
        self._update_timedelta = update_timedelta
        self._cached_output = None
        self._last_update_time = None

    def __call__(self, state, **kwargs):
        if ((self._last_update_time is None) or
                (state['time'] >= self._last_update_time +
                 self._update_timedelta)):
            self._cached_output = self._prognostic(state, **kwargs)
            self._last_update_time = state['time']
        return self._cached_output

    def __getattr__(self, item):
        return getattr(self._prognostic, item)


class TendencyInDiagnosticsWrapper(object):
    """
    Wraps a prognostic object so that when it is called, it returns all
    tendencies in its diagnostics.

    Example
    -------
    This how the wrapper should be used on a fictional Prognostic class
    called RRTMRadiation.

    >>> prognostic = TendencyInDiagnosticsWrapper(RRTMRadiation(), 'radiation')
    """

    def __init__(self, prognostic, label):
        """
        Initialize the Delayed object.

        Args
        ----
        prognostic : Prognostic
            The object to be wrapped
        label : str
            Label describing what the tendencies are due to, to be
            put in the diagnostic quantity names.
        """
        self._prognostic = prognostic
        self._tendency_label = label
        self._tendency_diagnostic_properties = {}
        for quantity_name, properties in prognostic.tendency_properties.items():
            print('setting property for {}'.format(quantity_name))
            diagnostic_name = 'tendency_of_{}_due_to_{}'.format(quantity_name, label)
            self._tendency_diagnostic_properties[diagnostic_name] = properties

    @property
    def diagnostics(self):
        return list(self.diagnostic_properties.keys())

    @property
    def diagnostic_properties(self):
        return_dict = self._prognostic.diagnostic_properties.copy()
        return_dict.update(self._tendency_diagnostic_properties)
        return return_dict

    def __call__(self, state, **kwargs):
        tendencies, diagnostics = self._prognostic(state, **kwargs)
        for quantity_name in tendencies.keys():
            diagnostic_name = 'tendency_of_{}_due_to_{}'.format(
                quantity_name, self._tendency_label)
            diagnostics[diagnostic_name] = tendencies[quantity_name]
        return tendencies, diagnostics

    def __getattr__(self, item):
        return getattr(self._prognostic, item)


def replace_none_with_default(constant_name, value):
    """If value is None, returns the default constant for the constant name.
    Otherwise, returns value. If the default constant is not defined, raises
    KeyError."""
    if value is None:
        return default_constants[constant_name]
    else:
        return value


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
            dict1[key] = dict2[key]
        else:
            if (isinstance(dict1[key], DataArray) and isinstance(dict2[key], DataArray) and
                    ('units' in dict1[key].attrs) and ('units' in dict2[key].attrs)):
                dict1[key] += dict2[key].to_units(dict1[key].attrs['units'])
            else:
                dict1[key] += dict2[key]  # += is in-place addition operator
    return  # not returning anything emphasizes that this is in-place


def ensure_no_shared_keys(dict1, dict2):
    """
    Raises SharedKeyException if there exists a key present in both
    dictionaries.
    """
    shared_keys = set(dict1.keys()).intersection(dict2.keys())
    if len(shared_keys) > 0:
        raise SharedKeyException(
            'unexpected shared keys: {}'.format(shared_keys))


def get_numpy_array(data_array, out_dims):
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
        :py:function:`~sympl.set_dimension_names`. '*' indicates an axis
        which is the flattened collection of all dimensions not explicitly
        listed in out_dims, including any dimensions with unknown direction.

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
    current_dim_names = dim_names.copy()
    for dim in out_dims:
        if dim not in ('x', 'y', 'z', '*'):
            current_dim_names[dim] = [dim]
    direction_to_names = get_input_array_dim_names(data_array, out_dims, current_dim_names)
    target_dimension_order = get_target_dimension_order(out_dims, direction_to_names)
    slices_or_none = get_slices_and_placeholder_nones(
        data_array, out_dims, direction_to_names)
    final_shape = get_final_shape(data_array, out_dims, direction_to_names)
    return np.reshape(data_array.transpose(
        *target_dimension_order).values[slices_or_none], final_shape)


def get_input_array_dim_names(data_array, out_dims, dim_names):
    """
    Takes in a DataArray and an iterable of directions
    ('x', 'y', 'z', or '*'). Returns a dictionary mapping those directions to
    a list of dimension names corresponding to those directions in data_array.
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
        elif (direction is not '*') and (len(direction_to_names[direction]) > 1):
            raise ValueError(
                'DataArray has multiple dimensions for a single direction')
        else:
            for name in direction_to_names[direction]:
                slices_or_none.append(slice(0, len(data_array.coords[name])))
    return slices_or_none


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
                np.product([len(data_array.coords[name])
                            for name in direction_to_names[direction]]))
    return final_shape


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
        :py:function:`~sympl.set_dimension_names`. '*' indicates an axis
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
    current_dim_names = dim_names.copy()
    for dim in from_dims:
        if dim not in ('x', 'y', 'z', '*'):
            current_dim_names[dim] = [dim]
    direction_to_names = get_input_array_dim_names(result_like, from_dims, current_dim_names)
    original_shape = []
    original_coords = []
    for direction in from_dims:
        if direction in direction_to_names.keys():
            for name in direction_to_names[direction]:
                original_shape.append(len(result_like.coords[name]))
                original_coords.append(result_like.coords[name])
    data_array = DataArray(
        np.reshape(array, original_shape), coords=original_coords).transpose(
            *list(result_like.coords))
    if result_attrs is not None:
        data_array.attrs = result_attrs
    return data_array
