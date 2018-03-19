from .base_components import ImplicitPrognostic
from .array import DataArray


class TimeDifferencingWrapper(ImplicitPrognostic):
    """
    Wraps an Implicit object and turns it into an ImplicitPrognostic by applying
    simple first-order time differencing to determine tendencies.

    Example
    -------
    This how the wrapper should be used on an Implicit class
    called GridScaleCondensation.

    >>> component = TimeDifferencingWrapper(GridScaleCondensation())
    """

    def __init__(self, implicit):
        self._implicit = implicit

    def __call__(self, state, timestep):
        diagnostics, new_state = self._implicit(state, timestep)
        tendencies = {}
        timestep_seconds = timestep.total_seconds()
        for varname, data_array in new_state.items():
            if isinstance(data_array, DataArray):
                if varname in self._implicit.output_properties.keys():
                    if varname not in state.keys():
                        raise RuntimeError(
                            'Cannot calculate tendency for {} because it is not'
                            ' present in the input state.'.format(varname))
                    tendency = (data_array - state[varname].to_units(data_array.attrs['units'])) / timestep_seconds
                    if data_array.attrs['units'] == '':
                        tendency.attrs['units'] = 's^-1'
                    else:
                        tendency.attrs['units'] = data_array.attrs['units'] + ' s^-1'
                    tendencies[varname] = tendency.to_units(
                        self._implicit.output_properties[varname]['units'] + ' s^-1')
            elif varname != 'time':
                raise ValueError(
                    'Wrapped implicit gave an output {} of type {}, but should'
                    'only give sympl.DataArray objects.'.format(
                        varname, type(data_array)))
        return tendencies, diagnostics

    @property
    def tendencies(self):
        return list(self.tendency_properties.keys())

    @property
    def tendency_properties(self):
        return_dict = self._implicit.output_properties.copy()
        return_dict.update(self._tendency_diagnostic_properties)
        return return_dict

    def __getattr__(self, item):
        if item not in ('outputs', 'output_properties'):
            return getattr(self._implicit, item)
