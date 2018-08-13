from .exceptions import InvalidPropertyDictError
import numpy as np
from .units import units_are_same
from .get_np_arrays import get_numpy_arrays_with_properties
from .restore_dataarray import restore_data_arrays_with_properties

_tracer_unit_dict = {}
_tracer_names = []
_packers = set()


def reset_tracers():
    global _tracer_names
    while len(_tracer_unit_dict) > 0:
        _tracer_unit_dict.popitem()
    while len(_tracer_names) > 0:
        _tracer_names.pop()


def reset_packers():
    while len(_packers) > 0:
        _packers.pop()


def register_tracer(name, units):
    """
    Parameters
    ----------
    name : str
        Quantity name to register as a tracer.
    units : str
        Unit string of that quantity.
    """
    if name in _tracer_unit_dict and not units_are_same(_tracer_unit_dict[name], units):
        raise ValueError(
            'Tracer {} is already registered with units {} which are different '
            'from input units {}'.format(
                name, _tracer_unit_dict[name], units
            )
        )
    _tracer_unit_dict[name] = units
    _tracer_names.append(name)
    for packer in _packers:
        packer.ensure_tracer_not_in_outputs(name, units)


def get_tracer_unit_dict():
    """
    Returns
    -------
    unit_dict : dict
        A dictionary whose keys are tracer quantity names as str and values are
        units of those quantities as str.
    """
    return_dict = {}
    return_dict.update(_tracer_unit_dict)
    return return_dict


def get_tracer_names():
    """
    Returns
    -------
    tracer_names : tuple of str
        Tracer names in the order that they will appear in tracer arrays.\
    """
    return tuple(_tracer_names)


def get_tracer_input_properties(prepend_tracers, tracer_dims):
    """

    Args:
        prepend_tracers (list of tuple): Pairs of (name, units) describing
            tracers that are to be included in addition to any registered
            tracers.
        tracer_dims (list): Dimensions to use for each tracer
            (e.g. ['dim1', 'dim2']).

    Returns:
        input_properties (dict): A properties dictionary for registered and
            additional tracers.
    """
    tracer_dims = list(tracer_dims)
    if 'tracer' in tracer_dims:
        tracer_dims.remove('tracer')
    tracer_units = {}
    tracer_units.update(get_tracer_unit_dict())
    tracer_units.update(dict(prepend_tracers))
    tracer_properties = {}
    for name, units in tracer_units.items():
        tracer_properties[name] = {'units': units, 'dims': tracer_dims}
    return tracer_properties


def get_quantity_dims(tracer_dims):
    if 'tracer' not in tracer_dims:
        raise ValueError("Tracer dims must include a dimension named 'tracer'")
    quantity_dims = list(tracer_dims)
    quantity_dims.remove('tracer')
    return tuple(quantity_dims)


class TracerPacker(object):

    def __init__(self, component, tracer_dims, prepend_tracers=None):
        self._prepend_tracers = prepend_tracers or ()
        self._tracer_dims = tuple(tracer_dims)
        self._tracer_quantity_dims = get_quantity_dims(tracer_dims)
        if hasattr(component, 'tendency_properties') or hasattr(component, 'output_properties'):
            self.component = component
        else:
            raise TypeError(
                'Expected a component object subclassing type Stepper, '
                'ImplicitTendencyComponent, or TendencyComponent but received component of '
                'type {}'.format(component.__class__.__name__))
        for name, units in self._prepend_tracers:
            if name not in _tracer_unit_dict.keys():
                self.ensure_tracer_not_in_outputs(name, units)
        for name, units in _tracer_unit_dict.items():
            self.ensure_tracer_not_in_outputs(name, units)
        _packers.add(self)

    def ensure_tracer_not_in_outputs(self, name, units):
        if hasattr(self.component, 'tendency_properties'):
            self._ensure_tracer_not_in_tendency_properties(name, units)
        elif hasattr(self.component, 'output_properties'):
            self.ensure_tracer_not_in_output_properties(name, units)

    def ensure_tracer_not_in_output_properties(self, name, units):
        if name in self.component.output_properties.keys():
            raise InvalidPropertyDictError(
                'Attempted to insert {} as tracer to component of type {} but '
                'it already has that quantity defined as an output.'.format(
                    name, self.component.__class__.__name__
                )
            )

    def _ensure_tracer_not_in_tendency_properties(self, name, units):
        if name in self.component.tendency_properties.keys():
            raise InvalidPropertyDictError(
                'Attempted to insert {} as tracer to component of type {} but '
                'it already has that quantity defined as a tendency '
                'output.'.format(
                    name, self.component.__class__.__name__
                )
            )

    def _ensure_tracer_quantity_dims(self, dims):
        if tuple(self._tracer_quantity_dims) != tuple(dims):
            raise InvalidPropertyDictError(
                'Tracers have conflicting dims {} and {}'.format(
                    self._tracer_quantity_dims, dims)
            )

    def register_tracers(self, unit_dict, input_properties):
        for name, units in unit_dict.items():
            if name not in input_properties.keys():
                input_properties[name] = {
                    'dims': self._tracer_quantity_dims,
                    'units': units,
                    'tracer': True,
                }

    @property
    def tracer_names(self):
        return_list = []
        for name, units in self._prepend_tracers:
            return_list.append(name)
        for name in _tracer_names:
            if name not in return_list:
                return_list.append(name)
        return tuple(return_list)

    @property
    def _tracer_index(self):
        return self._tracer_dims.index('tracer')

    def pack(self, state):
        """

        Args:
            state (dict): A state dictionary.

        Returns:
            tracer_array (ndarray): An array containing the tracer data, with
                dimensions as specified by tracer_dims on initializing this
                object.
        """
        tracer_properties = get_tracer_input_properties(
            self._prepend_tracers, self._tracer_quantity_dims)
        raw_state = get_numpy_arrays_with_properties(state, tracer_properties)
        if len(self.tracer_names) == 0:
            shape = [0 for dim in self._tracer_dims]
        else:
            shape = list(raw_state[self.tracer_names[0]].shape)
            shape.insert(self._tracer_index, len(self.tracer_names))
        array = np.empty(shape, dtype=np.float64)
        for i, name in enumerate(self.tracer_names):
            tracer_slice = [slice(0, d) for d in shape]
            tracer_slice[self._tracer_index] = i
            array[tracer_slice] = raw_state[name]
        return array

    def unpack(self, tracer_array, input_state, multiply_unit=''):
        """

        Args:
            tracer_array (ndarray): An array containing tracer values, with
                dimensions as specified by tracer_dims on initializing this
                object.
            input_state (dict): A state dictionary from which the tracer array
                was originally packed.
            multiply_unit (str, optional): A unit string which should be multiplied to
                the units of each output in the returned DataArrays, for example
                to represent the units over which a time difference was taken.

        Returns:
            tracer_dict (dict): A dictionary whose keys are tracer names and
                values are DataArrays containing the values of each
                tracer.
        """
        tracer_properties = get_tracer_input_properties(
            self._prepend_tracers, self._tracer_quantity_dims)
        raw_state = {}
        for i, name in enumerate(self.tracer_names):
            tracer_slice = [slice(0, d) for d in tracer_array.shape]
            tracer_slice[self._tracer_index] = i
            raw_state[name] = tracer_array[tracer_slice]
        out_properties = {}
        for name, properties in tracer_properties.items():
            out_properties[name] = properties.copy()
            if multiply_unit is not '':
                out_properties[name]['units'] = '{} {}'.format(
                    out_properties[name]['units'], multiply_unit)
        return_state = restore_data_arrays_with_properties(
            raw_state, out_properties, input_state, tracer_properties)
        return return_state
