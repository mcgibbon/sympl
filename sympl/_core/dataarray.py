import xarray as xr
from pint.errors import DimensionalityError
from .units import data_array_to_array_with_units


class DataArray(xr.DataArray):

    def __add__(self, other):
        """If this DataArray is on the left side of the addition, keep its
        attributes when adding to the other object."""
        result = super(DataArray, self).__add__(other)
        result.attrs = self.attrs
        return result

    def __sub__(self, other):
        """If this DataArray is on the left side of the subtraction, keep its
        attributes when subtracting the other object."""
        result = super(DataArray, self).__sub__(other)
        result.attrs = self.attrs
        return result

    def to_units(self, units):
        """
        Convert the units of this DataArray, if necessary. No conversion is
        performed if the units are the same as the units of this DataArray.
        The units of this DataArray are determined from the "units" attribute in
        attrs.

        Args
        ----
        units : str
            The desired units.

        Raises
        ------
        ValueError
            If the units are invalid for this object.
        KeyError
            If this object does not have units information in its attrs.

        Returns
        -------
        converted_data : DataArray
            A DataArray containing the data from this object in the
            desired units, if possible.
        """
        if 'units' not in self.attrs:
            raise KeyError('"units" not present in attrs')
        try:
            out_array = data_array_to_array_with_units(self, units)
            out_attrs = {}
            out_attrs.update(self.attrs)
            out_attrs['units'] = units
            return DataArray(
                out_array, dims=self.dims, coords=self.coords, attrs=out_attrs
            )
        except DimensionalityError as err:
            raise ValueError(str(err))
