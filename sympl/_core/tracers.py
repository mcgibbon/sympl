from .exceptions import InvalidPropertyDictError
import numpy as np
from .units import units_are_same

_tracer_unit_dict = {}
_tracer_names = []
_packers = set()


def clear_tracers():
    global _tracer_names
    while len(_tracer_unit_dict) > 0:
        _tracer_unit_dict.popitem()
    while len(_tracer_names) > 0:
        _tracer_names.pop()


def clear_packers():
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
        packer.insert_tracer_to_properties(name, units)


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
                'Expected a component object subclassing type Implicit, '
                'ImplicitPrognostic, or Prognostic but received component of '
                'type {}'.format(component.__class__.__name__))
        for name, units in self._prepend_tracers:
            if name not in _tracer_unit_dict.keys():
                self.insert_tracer_to_properties(name, units)
        for name, units in _tracer_unit_dict.items():
            self.insert_tracer_to_properties(name, units)
        _packers.add(self)

    def insert_tracer_to_properties(self, name, units):
        self._insert_tracer_to_input_properties(name, units)
        if hasattr(self.component, 'tendency_properties'):
            self._insert_tracer_to_tendency_properties(name, units)
        elif hasattr(self.component, 'output_properties'):
            self._insert_tracer_to_output_properties(name, units)

    def _insert_tracer_to_input_properties(self, name, units):
        if name in self.component.input_properties.keys():
            raise InvalidPropertyDictError(
                'Attempted to insert {} as tracer to component of type {} but '
                'it already has that quantity defined as an input.'.format(
                    name, self.component.__class__.__name__
                )
            )
        if name not in self.component.input_properties:
            self.component.input_properties[name] = {
                'dims': self._tracer_quantity_dims,
                'units': units,
                'tracer': True,
            }

    def _insert_tracer_to_output_properties(self, name, units):
        if name in self.component.output_properties.keys():
            raise InvalidPropertyDictError(
                'Attempted to insert {} as tracer to component of type {} but '
                'it already has that quantity defined as an output.'.format(
                    name, self.component.__class__.__name__
                )
            )
        if name not in self.component.output_properties:
            self.component.output_properties[name] = {
                'dims': self._tracer_quantity_dims,
                'units': units,
                'tracer': True,
            }

    def _insert_tracer_to_tendency_properties(self, name, units):
        time_unit = getattr(self.component, 'tracer_tendency_time_unit', 's')
        if name in self.component.tendency_properties.keys():
            raise InvalidPropertyDictError(
                'Attempted to insert {} as tracer to component of type {} but '
                'it already has that quantity defined as a tendency '
                'output.'.format(
                    name, self.component.__class__.__name__
                )
            )
        if name not in self.component.tendency_properties:
            self.component.tendency_properties[name] = {
                'dims': self._tracer_quantity_dims,
                'units': '{} {}^-1'.format(units, time_unit),
                'tracer': True,
            }

    def is_tracer(self, tracer_name):
        return self.component.input_properties.get(tracer_name, {}).get('tracer', False)

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
            if name not in return_list and self.is_tracer(name):
                return_list.append(name)
        return tuple(return_list)

    @property
    def _tracer_index(self):
        return self._tracer_dims.index('tracer')

    def pack(self, raw_state):
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

    def unpack(self, tracer_array):
        return_state = {}
        for i, name in enumerate(self.tracer_names):
            tracer_slice = [slice(0, d) for d in tracer_array.shape]
            tracer_slice[self._tracer_index] = i
            return_state[name] = tracer_array[tracer_slice]
        return return_state
