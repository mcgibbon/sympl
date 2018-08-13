from .exceptions import InvalidPropertyDictError
from .tracers import get_tracer_input_properties
from .units import units_are_compatible


def combine_dims(dims1, dims2):
    """
    Takes in two dims specifications and returns a single specification that
    satisfies both, if possible. Raises an InvalidPropertyDictError if not.

    Parameters
    ----------
    dims1 : iterable of str
    dims2 : iterable of str

    Returns
    -------
    dims : iterable of str

    Raises
    ------
    InvalidPropertyDictError
        If the two dims specifications cannot be combined
    """
    if dims1 == dims2:
        return dims1
    dims_out = []
    dims1 = set(dims1)
    dims2 = set(dims2)
    dims1_wildcard = '*' in dims1
    dims1.discard('*')
    dims2_wildcard = '*' in dims2
    dims2.discard('*')
    unmatched_dims = set(dims1).union(dims2).difference(dims_out)
    shared_dims = set(dims2).intersection(dims2)
    if dims1_wildcard and dims2_wildcard:
        dims_out.insert(0, '*')  # either dim can match anything
        dims_out.extend(unmatched_dims)
    elif not dims1_wildcard and not dims2_wildcard:
        if shared_dims != set(dims1) or shared_dims != set(dims2):
            raise InvalidPropertyDictError(
                'dims {} and {} are incompatible'.format(dims1, dims2))
        dims_out.extend(unmatched_dims)
    elif dims1_wildcard:
        if shared_dims != set(dims2):
            raise InvalidPropertyDictError(
                'dims {} and {} are incompatible'.format(dims1, dims2))
        dims_out.extend(unmatched_dims)
    elif dims2_wildcard:
        if shared_dims != set(dims1):
            raise InvalidPropertyDictError(
                'dims {} and {} are incompatible'.format(dims1, dims2))
        dims_out.extend(unmatched_dims)
    return dims_out


def combine_component_properties(component_list, property_name, input_properties=None):
    property_list = []
    for component in component_list:
        property_list.append(getattr(component, property_name))
        if property_name == 'input_properties' and getattr(component, 'uses_tracers', False):
            tracer_dims = list(component.tracer_dims)
            if 'tracer' not in tracer_dims:
                raise InvalidPropertyDictError(
                    "tracer_dims must include a 'tracer' dimension indicating "
                    "tracer number"
                )
            tracer_dims.remove('tracer')
            property_list.append(get_tracer_input_properties(getattr(component, 'prepend_tracers', ()), tracer_dims))
    return combine_properties(property_list, input_properties)


def combine_properties(property_list, input_properties=None):
    if input_properties is None:
        input_properties = {}
    return_dict = {}
    for property_dict in property_list:
        for name, properties in property_dict.items():
            if name not in return_dict:
                return_dict[name] = {}
                return_dict[name].update(properties)
                if 'dims' not in properties.keys():
                    if name in input_properties.keys() and 'dims' in input_properties[name].keys():
                        return_dict[name]['dims'] = input_properties[name]['dims']
                    else:
                        raise InvalidPropertyDictError()
            elif not units_are_compatible(
                    properties['units'], return_dict[name]['units']):
                raise InvalidPropertyDictError(
                    'Cannot combine components with incompatible units '
                    '{} and {} for quantity {}'.format(
                        return_dict[name]['units'],
                        properties['units'], name))
            else:
                if 'dims' in properties.keys():
                    new_dims = properties['dims']
                elif name in input_properties.keys() and 'dims' in input_properties[name].keys():
                    new_dims = input_properties[name]['dims']
                else:
                    raise InvalidPropertyDictError()
                try:
                    dims = combine_dims(return_dict[name]['dims'], new_dims)
                    return_dict[name]['dims'] = dims
                except InvalidPropertyDictError as err:
                    raise InvalidPropertyDictError(
                        'Incompatibility between dims of quantity {}: {}'.format(
                            name, err.args[0]))
    return return_dict
