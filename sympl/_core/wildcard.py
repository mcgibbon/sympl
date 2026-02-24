import numpy as np

from .backend import get_backend
from .exceptions import InvalidPropertyDictError, InvalidStateError


def get_wildcard_matches_and_dim_lengths(state, property_dictionary):
    wildcard_names = []
    dim_lengths = {}
    backend = get_backend()
    # Loop to get the set of names matching "*" (wildcard names)
    for quantity_name, properties in property_dictionary.items():
        ensure_properties_have_dims_and_units(properties, quantity_name)
        value_dims = backend.get_dims(state[quantity_name])
        value_shape = backend.get_shape(state[quantity_name])
        for dim_name, length in zip(value_dims, value_shape):
            if dim_name not in dim_lengths.keys():
                dim_lengths[dim_name] = length
            elif length != dim_lengths[dim_name]:
                raise InvalidStateError(
                    "Dimension {} conflicting lengths {} and {} in different "
                    "state quantities.".format(dim_name, length, dim_lengths[dim_name])
                )
        new_wildcard_names = [
            dim for dim in value_dims if dim not in properties["dims"]
        ]
        if len(new_wildcard_names) > 0 and "*" not in properties["dims"]:
            raise InvalidStateError(
                "Quantity {} has unexpected dimensions {}.".format(
                    quantity_name, new_wildcard_names
                )
            )
        wildcard_names.extend(
            [name for name in new_wildcard_names if name not in wildcard_names]
        )
    if not any(
        "dims" in p.keys() and "*" in p["dims"] for p in property_dictionary.values()
    ):
        wildcard_names = (
            None  # can't determine wildcard matches if there is no wildcard
        )
    else:
        wildcard_names = tuple(wildcard_names)
    return wildcard_names, dim_lengths


def flatten_wildcard_dims(array, i_start, i_end):
    if i_end > len(array.shape):
        raise ValueError("i_end should be less than the number of axes in array")
    elif i_start < 0:
        raise ValueError("i_start should be greater than 0")
    elif i_start > i_end:
        raise ValueError("i_start should be less than or equal to i_end")
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


def fill_dims_wildcard(out_dims, dim_lengths, wildcard_names, expand_wildcard=True):
    i_wildcard = out_dims.index("*")
    target_shape = []
    out_dims_without_wildcard = []
    for i, out_dim in enumerate(out_dims):
        if i == i_wildcard and expand_wildcard:
            target_shape.extend([dim_lengths[n] for n in wildcard_names])
            out_dims_without_wildcard.extend(wildcard_names)
        elif i == i_wildcard and not expand_wildcard:
            target_shape.append(np.prod([dim_lengths[n] for n in wildcard_names]))
        else:
            target_shape.append(dim_lengths[out_dim])
            out_dims_without_wildcard.append(out_dim)
    return out_dims_without_wildcard, target_shape


def expand_array_wildcard_dims(raw_array, target_shape, name, out_dims):
    try:
        out_array = np.reshape(raw_array, target_shape)
    except ValueError:
        raise InvalidPropertyDictError(
            "Failed to restore shape for output {} with raw shape {} "
            "and target shape {}, are the output dims {} correct?".format(
                name, raw_array.shape, target_shape, out_dims
            )
        )
    return out_array


def ensure_properties_have_dims_and_units(properties, quantity_name):
    if "dims" not in properties:
        raise InvalidPropertyDictError(
            "dims not specified for quantity {}".format(quantity_name)
        )
    if "units" not in properties:
        raise InvalidPropertyDictError(
            "units not specified for quantity {}".format(quantity_name)
        )
