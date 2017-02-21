from .exceptions import SharedKeyException
from .constants import default_constants
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


def put_prognostic_tendency_in_diagnostics(prognostic_class, label):
    """
    Wraps the class prognostic_class so that when it is called, its
    tendencies are all put in its diagnostics dictionary as
    "tendency_of_{quantity}_due_to_{label}", where label is passed in to this
    function.

    Note that a *class* is wrapped and returned. You must afterwards
    instantiate that class if you want to use it.

    Example
    -------
    This how the function should be used on a Prognostic class RRTMRadiation.

    >>> RRTMRadiation = put_prognostic_tendency_in_diagnostics(RRTMRadiation, 'radiation')
    >>> prognostic = RRTMRadiation()

    Args
    ----
    prognostic_class : type
        A subclass of Prognostic (not an instance of that class).
    label : str
        Label describing what the tendencies are due to, to be
        put in the diagnostic quantity names.

    Returns
    -------
    WrappedPrognostic : type
        The subclass of Prognostic wrapped as described above.
    """
    class WrappedPrognostic(prognostic_class):
        _pptid_tendency_label = label

        def __init__(self, *args, **kwargs):
            super(WrappedPrognostic, self).__init__(*args, **kwargs)
            diagnostic_names = list(self.diagnostics)
            for quantity_name in self.tendencies:
                diagnostic_names.append('tendency_of_{}_due_to_{}'.format(
                    quantity_name, self._pptid_tendency_label))
            self.diagnostics = tuple(diagnostic_names)

        def __call__(self, state, **kwargs):
            tendencies, diagnostics = super(WrappedPrognostic, self).__call__(
                state, **kwargs)
            for quantity_name in tendencies.keys():
                diagnostic_name = 'tendency_of_{}_due_to_{}'.format(
                    quantity_name, self._pptid_tendency_label)
                diagnostics[diagnostic_name] = tendencies[quantity_name]
            return tendencies, diagnostics

    return WrappedPrognostic


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


def get_numpy_array(data_array, out_dims):
    """
    Retrieve a numpy array with the desired dimensions and dimension order
    from the given DataArray, using transpose and creating length 1 dimensions
    as necessary.

    Args
    ----
    data_array : DataArray
        The object from which to retrieve data.
    out_dims : {'x', 'y', 'z'}
        The desired dimensions of the output and their order.
        Length 1 dimensions will be created if the dimension
        does not exist in data_array. Values in the iterable should be 'x',
        'y', or 'z'.

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
