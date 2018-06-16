from .exceptions import InvalidPropertyDictError
import numpy as np

_tracer_unit_dict = {}


def clear_tracer_unit_dict():
    while len(_tracer_unit_dict) > 0:
        _tracer_unit_dict.pop()


def register_tracer(name, units):
    _tracer_unit_dict[name] = units


def get_tracer_unit_dict():
    return_dict = {}
    return_dict.update(_tracer_unit_dict)
    return return_dict


def get_quantity_dims(tracer_dims):
    if 'tracer' not in tracer_dims:
        raise ValueError("Tracer dims must include a dimension named 'tracer'")
    quantity_dims = list(tracer_dims)
    quantity_dims.remove('tracer')
    return tuple(quantity_dims)


class TracerPacker(object):

    def __init__(self, component, tracer_dims):
        self.tracer_names = []
        self._tracer_dims = tuple(tracer_dims)
        self._tracer_quantity_dims = get_quantity_dims(tracer_dims)
        if component is not None:
            for name, properties in component.input_properties.items():
                if properties.get('tracer'):
                    self._ensure_tracer_quantity_dims(properties['dims'])
                    self.tracer_names.append(name)

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
    def _tracer_index(self):
        return self._tracer_dims.index('tracer')

    def pack(self, raw_state):
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
