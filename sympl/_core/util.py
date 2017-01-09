from .exceptions import SharedKeyException
from .constants import default_constants
from six import string_types
try:
    from numba import jit
except ImportError:
    # define a function with the same call signature as jit that does nothing
    def jit(signature_or_function=None, **kwargs):
        if signature_or_function is None:
            return lambda x: x
        else:
            return signature_or_function

dim_names = {'x': [], 'y': [], 'z': []}


def set_dimension_names(x=None, y=None, z=None):
    for key, value in [('x', x), ('y', y), ('z', z)]:
        if isinstance(value, string_types):
            dim_names[key] = [key, value]
        else:
            dim_names[key] = [key] + list(value)


def combine_dimensions(arrays, out_dims):
    """
    Returns a tuple of dimension names corresponding to
    dimension names from the DataArray objects given by *args when present.
    The names returned correspond to the directions in out_dims.

    Args:
        arrays (iterable of DataArray): Objects for which to combine
            dimension names.
        out_dims (iterable of str): The desired output directions. Should
            contain only 'x', 'y', or 'z'. For example, ('y', 'x') is valid.

    Raises:
        ValueError: If there are multiple names for a single direction, or if
            an array has a dimension along a direction not present in out_dims.

    Returns:
        dimensions (list of str): The combined dimension names, in the order
            given by out_dims.
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


def set_prognostic_update_frequency(prognostic_class, update_timedelta):
    """
    Alters a prognostic class so that when it is called, it only computes its
    output once for every period of length update_timedelta. In between these
    calls, the cached output from the last computation will be returned.

    Note that the *class* itself must be updated, not an *instance* of that
    class.

    Once modified, the class requires that the 'time' quantity is set on
    states it receives, and that it is a datetime or timedelta object.

    Example:
        This how the function should be used on a Prognostic class MyPrognostic.

        >>> from datetime import timedelta
        >>> set_prognostic_update_frequency(MyPrognostic, timedelta(hours=1))
        >>> prognostic = MyPrognostic()

    Args:
        prognostic_class (type): A Prognostic class (not an instance).
        update_timedelta (timedelta): The amount that state['time'] must differ
            from when output was cached before new output is
            computed.
    """
    prognostic_class._spuf_update_timedelta = update_timedelta
    prognostic_class._spuf_last_update_time = None
    original_call = prognostic_class.__call__

    def __call__(self, state):
        if (self._spuf_last_update_time is None or
                state['time'] >= self._spuf_last_update_time + self._spuf_update_timedelta):
            self._spuf_cached_output = original_call(self, state)
            self._spuf_last_update_time = state['time']
        return self._spuf_cached_output

    prognostic_class.__call__ = __call__


def put_prognostic_tendency_in_diagnostics(prognostic_class, label):
    """
    Modifies the class prognostic_class so that when it is called, its
    tendencies are all put in its diagnostics as
    "tendency_of_{quantity}_due_to_{label}", where label is passed in to this
    function.

    Example:
        This how the function should be used on a Prognostic class RRTMRadiation.

        >>> put_prognostic_tendency_in_diagnostics(RRTMRadiation, 'radiation')
        >>> prognostic = RRTMRadiation()

    Args:
        prognostic_class (type): A Prognostic class (not an instance).
        label (str): Label describing what the tendencies are due to, to be
            put in the diagnostic quantity names.
    """
    prognostic_class._pptid_tendency_label = label
    original_call = prognostic_class.__call__

    def __call__(self, state):
        tendencies, diagnostics = original_call(self, state)
        for quantity_name in tendencies.keys():
            diagnostic_name = 'tendency_of_{}_due_to_{}'.format(
                quantity_name, self._pptid_tendency_label)
            diagnostics[diagnostic_name] = tendencies[quantity_name]
        return tendencies, diagnostics

    prognostic_class.__call__ = __call__


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
    array-like, to avoid data copying.
    """
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = dict2[key]
        else:
            try:
                # works for array-like objects, in-place
                dict1[key][:] += dict2[key][:]
            except TypeError:
                dict1[key] += dict2[key]
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


def get_numpy_array(data_array, out_dims=('x', 'y', 'z')):
    """
    Retrieve a numpy array with the desired dimensions and dimension order
    from the given DataArray, using transpose and creating length 1 dimensions
    as necessary.

    Args:
        data_array (DataArray): The object from which to retrieve data.
        out_dims (iterable of str): The desired dimensions of the output and
            their order. Length 1 dimensions will be created if the dimension
            does not exist in data_array. Values in the iterable should be 'x',
            'y', or 'z'.

    Returns:
        numpy_array (ndarray): The desired array, with dimensions in the
            correct order and length 1 dimensions created as needed.
    """
    indices = [None for _ in range(len(out_dims))]
    dimensions = []
    for i, dimension_names in zip(
            range(len(out_dims)), [dim_names[name] for name in out_dims]):
        dims = set(data_array.dims).intersection(dimension_names)
        if len(dims) == 1:
            dim = dims.pop()
            dimensions.append(dim)
            indices[i] = slice(0, len(data_array.coords[dim]))
        elif len(dims) > 1:
            raise ValueError(
                'DataArray has multiple dimensions for a single direction')
    if len(dimensions) < len(data_array.dims):
        raise ValueError(
            'Was not able to classify all dimensions: {}'.format(
                data_array.dims))
    # Transpose correctly orders existing dimensions
    # indices creates new length-1 dimensions
    return data_array.transpose(*dimensions).values[indices]
