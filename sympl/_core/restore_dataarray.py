import numpy as np

from .backend import get_backend
from .dataarray import DataArray
from .exceptions import InvalidPropertyDictError
from .wildcard import (
    expand_array_wildcard_dims,
    fill_dims_wildcard,
    get_wildcard_matches_and_dim_lengths,
)


def ensure_values_are_arrays(array_dict):
    for name, value in array_dict.items():
        if not isinstance(value, np.ndarray):
            array_dict[name] = np.asarray(value)


def get_alias_or_name(name, output_properties, input_properties):
    if "alias" in output_properties[name].keys():
        raw_name = output_properties[name]["alias"]
    elif name in input_properties.keys() and "alias" in input_properties[name].keys():
        raw_name = input_properties[name]["alias"]
    else:
        raw_name = name
    return raw_name


def check_array_shape(out_dims, raw_array, name, dim_lengths):
    if len(out_dims) != len(raw_array.shape):
        raise InvalidPropertyDictError(
            "Returned array for {} has shape {} "
            "which is incompatible with dims {} in properties".format(
                name, raw_array.shape, out_dims
            )
        )
    for dim, length in zip(out_dims, raw_array.shape):
        if dim in dim_lengths.keys() and dim_lengths[dim] != length:
            raise InvalidPropertyDictError(
                "Dimension {} of quantity {} has length {}, but "
                "another quantity has length {}".format(
                    dim, name, length, dim_lengths[dim]
                )
            )


def restore_data_arrays_with_properties(
    raw_arrays,
    output_properties,
    input_state,
    input_properties,
    ignore_names=None,
    ignore_missing=False,
):
    """
    Parameters
    ----------
    raw_arrays : dict
        A dictionary whose keys are quantity names and values are numpy arrays
        containing the data for those quantities.
    output_properties : dict
        A dictionary whose keys are quantity names and values are dictionaries
        with properties for those quantities. The property "dims" must be
        present for each quantity not also present in input_properties. All
        other properties are included as attributes on the output DataArray
        for that quantity, including "units" which is required.
    input_state : dict
        A state dictionary that was used as input to a component for which
        DataArrays are being restored.
    input_properties : dict
        A dictionary whose keys are quantity names and values are dictionaries
        with input properties for those quantities. The property "dims" must be
        present, indicating the dimensions that the quantity was transformed to
        when taken as input to a component.
    ignore_names : iterable of str, optional
        Names to ignore when encountered in output_properties, will not be
        included in the returned dictionary.
    ignore_missing : bool, optional
        If True, ignore any values in output_properties not present in
        raw_arrays rather than raising an exception. Default is False.

    Returns
    -------
    out_dict : dict
        A dictionary whose keys are quantities and values are DataArrays
        corresponding to those quantities, with data, shapes and attributes
        determined from the inputs to this function.

    Raises
    ------
    InvalidPropertyDictError
        When an output property is specified to have dims_like an input
        property, but the arrays for the two properties have incompatible
        shapes.
    """
    raw_arrays = raw_arrays.copy()
    if ignore_names is None:
        ignore_names = []
    if ignore_missing:
        ignore_names = (
            set(output_properties.keys())
            .difference(raw_arrays.keys())
            .union(ignore_names)
        )
    wildcard_names, dim_lengths = get_wildcard_matches_and_dim_lengths(
        input_state, input_properties
    )
    ensure_values_are_arrays(raw_arrays)
    dims_from_out_properties = extract_output_dims_properties(
        output_properties, input_properties, ignore_names
    )
    out_dict = {}
    for name, out_dims in dims_from_out_properties.items():
        if name in ignore_names:
            continue
        raw_name = get_alias_or_name(name, output_properties, input_properties)
        if "*" in out_dims:
            for dim_name, length in zip(out_dims, raw_arrays[raw_name].shape):
                if dim_name not in dim_lengths and dim_name != "*":
                    dim_lengths[dim_name] = length
            out_dims_without_wildcard, target_shape = fill_dims_wildcard(
                out_dims, dim_lengths, wildcard_names
            )
            out_array = expand_array_wildcard_dims(
                raw_arrays[raw_name], target_shape, name, out_dims
            )
        else:
            check_array_shape(out_dims, raw_arrays[raw_name], name, dim_lengths)
            out_dims_without_wildcard = out_dims
            out_array = raw_arrays[raw_name]
        out_dict[name] = get_backend().create_quantity(
            out_array,
            name=name,
            dims=out_dims_without_wildcard,
            units=output_properties[name]["units"],
            reference_state=input_state,
        )
    return out_dict


def extract_output_dims_properties(output_properties, input_properties, ignore_names):
    return_array = {}
    for name, properties in output_properties.items():
        if name in ignore_names:
            continue
        elif "dims" in properties.keys():
            return_array[name] = properties["dims"]
        elif name not in input_properties.keys():
            raise InvalidPropertyDictError(
                "Output dims must be specified for {} in properties".format(name)
            )
        elif "dims" not in input_properties[name].keys():
            raise InvalidPropertyDictError(
                "Input dims must be specified for {} in properties".format(name)
            )
        else:
            return_array[name] = input_properties[name]["dims"]
    return return_array
