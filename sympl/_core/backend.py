import abc

import numpy as np

from .dataarray import DataArray
from .exceptions import InvalidStateError


class StateBackend(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_array(self, state_value, name, target_units, target_dims, dim_lengths):
        """
        Extract a raw array (numpy, jax, etc.) from a state value,
        converting units and aligning dimensions as requested.

        Args:
            state_value: The value from the state dictionary.
            name (str): The name of the quantity.
            target_units (str): The desired units.
            target_dims (tuple): The desired dimensions.
            dim_lengths (dict): Dictionary of dimension lengths for wildcard handling.

        Returns:
            array: A numpy-like array matching the target specs.
        """
        pass

    @abc.abstractmethod
    def create_quantity(self, data, name, units, dims, reference_state=None):
        """
        Wrap a raw output array into the backend's preferred container
        (e.g., DataArray, unyt_array, or JaxVariable).

        Args:
            data: The raw data array.
            name (str): The name of the quantity.
            units (str): The units of the quantity.
            dims (tuple): The dimensions of the quantity.
            reference_state (dict, optional): State to use for reference attributes/metadata.

        Returns:
            quantity: The wrapped quantity.
        """
        pass

    @abc.abstractmethod
    def get_dims(self, state_value):
        """
        Returns the dimensions of the state value as a tuple of strings.
        """
        pass

    @abc.abstractmethod
    def get_shape(self, state_value):
        """
        Returns the shape of the state value as a tuple of integers.
        """
        pass


class DataArrayBackend(StateBackend):
    """
    Default backend for Sympl, using xarray DataArrays and Pint for units.
    """

    def get_array(self, state_value, name, target_units, target_dims, dim_lengths):
        self._ensure_quantity_has_units(state_value, name)
        try:
            quantity = state_value.to_units(target_units)
        except ValueError:
            raise InvalidStateError(
                "Could not convert quantity {} from units {} to units {}".format(
                    name, state_value.attrs["units"], target_units
                )
            )

        return self._get_numpy_array(quantity, target_dims, dim_lengths)

    def create_quantity(self, data, name, units, dims, reference_state=None):
        return DataArray(data, dims=dims, attrs={"units": units})

    def get_dims(self, state_value):
        return state_value.dims

    def get_shape(self, state_value):
        return state_value.shape

    def _ensure_quantity_has_units(self, quantity, quantity_name):
        if "units" not in quantity.attrs:
            raise InvalidStateError(
                "quantity {} is missing units attribute".format(quantity_name)
            )

    def _get_numpy_array(self, data_array, out_dims, dim_lengths):
        """
        Gets a numpy array from the data_array with the desired out_dims, and a
        dict of dim_lengths that will give the length of any missing dims in the
        data_array.
        """
        values = data_array.values
        if len(values.shape) == 0 and len(out_dims) == 0:
            return values  # special case, 0-dimensional scalar array

        current_dims = list(data_array.dims)
        if current_dims == out_dims:
            return values

        missing_dims = [dim for dim in out_dims if dim not in current_dims]
        if missing_dims:
            # expand_dims in xarray adds the dimension at axis 0 by default.
            # Doing this sequentially means they are stacked at the front in reverse order.
            new_shape = (1,) * len(missing_dims) + values.shape
            values = values.reshape(new_shape)
            current_dims = missing_dims[::-1] + current_dims

        # Determine extra dims that are not in out_dims, and append them to target order
        # to mimic xarray.transpose behavior (it keeps unlisted dims at the end).
        extra_dims = [dim for dim in current_dims if dim not in out_dims]
        target_dims = list(out_dims) + extra_dims

        if current_dims == target_dims:
            numpy_array = values
        else:
            # Calculate transpose axes
            dim_to_axis = {dim: i for i, dim in enumerate(current_dims)}
            try:
                axes = [dim_to_axis[dim] for dim in target_dims]
            except KeyError:
                # This should be caught by missing_dims logic but just in case
                raise ValueError(
                    "Dimensions mismatch: target {} vs current {}".format(
                        target_dims, current_dims
                    )
                )
            numpy_array = values.transpose(axes)

        if not missing_dims:
            out_array = numpy_array
        else:
            # expand out missing dims which are currently length 1.
            # Construct out_shape carefully to handle extra dimensions from numpy_array
            base_out_shape = [dim_lengths.get(name, 1) for name in out_dims]
            # Append shapes of extra dimensions from numpy_array
            extra_shape = list(numpy_array.shape[len(out_dims) :])
            out_shape = base_out_shape + extra_shape

            if out_shape == list(numpy_array.shape):
                out_array = numpy_array
            else:
                out_array = np.empty(out_shape, dtype=numpy_array.dtype)
                out_array[:] = numpy_array
        return out_array


_current_backend = DataArrayBackend()


def get_backend():
    return _current_backend


def set_backend(backend):
    global _current_backend
    if not isinstance(backend, StateBackend):
        raise TypeError("Backend must inherit from StateBackend")
    _current_backend = backend
