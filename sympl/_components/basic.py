from .._core.array import DataArray
from .._core.base_components import ImplicitPrognostic, Prognostic, Diagnostic
from .._core.units import unit_registry as ureg


class ConstantPrognostic(Prognostic):
    """
    Prescribes constant tendencies provided at initialization.

    Note: Any arrays in the passed dictionaries are not copied, so that
        if you were to modify them after passing them into this object,
        it would also modify the values inside this object.
    """

    def __init__(self, tendencies, diagnostics=None):
        """
        Args
        ----
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second to be returned by this Prognostic.
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            to be returned by this Prognostic.
        """
        self._tendencies = tendencies.copy()
        if diagnostics is not None:
            self._diagnostics = diagnostics.copy()
        else:
            self._diagnostics = {}

    def __call__(self, state):
        """
        Gets tendencies and diagnostics from the passed model state. The
        returned dictionaries will contain the same values as were passed at
        initialization.

        Args
        ----
        state : dict
            A model state dictionary.

        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second.
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities.
        """
        return self._tendencies.copy(), self._diagnostics.copy()


class ConstantDiagnostic(Diagnostic):
    """
    Yields constant diagnostics provided at initialization.

    Note
    ----
    Any arrays in the passed dictionaries are not copied, so that
    if you were to modify them after passing them into this object,
    it would also modify the values inside this object.
    """

    def __init__(self, diagnostics):
        """
        Args
        ----
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities.
            The values in the dictionary will be returned when this
            Diagnostic is called.
        """
        self._diagnostics = diagnostics.copy()

    def __call__(self, state):
        """
        Returns diagnostic values.

        Args
        ----
        state : dict
            A model state dictionary. Is not used, and is only
            taken in to keep an API consistent with a Diagnostic.

        Returns
        -------
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities.
            The values in the returned dictionary are the same as were
            passed into this object at initialization.
        """
        return self._diagnostics.copy()


class RelaxationPrognostic(Prognostic):
    r"""
    Applies Newtonian relaxation to a single quantity.

    The relaxation takes the form
    :math:`\frac{dx}{dt} = - \frac{x - x_{eq}}{\tau}`
    where :math:`x` is the quantity being relaxed, :math:`x_{eq}` is the
    equilibrium value, and :math:`\tau` is the timescale of the relaxation.
    """

    def __init__(self, quantity_name, equilibrium_value=None,
                 relaxation_timescale=None):
        """
        Args
        ----
        quantity_name : str
            The name of the quantity to which Newtonian
            relaxation should be applied
        equilibrium_value : DataArray, optional
            The equilibrium value to which the quantity is relaxed. If
            not given, it should be provided in the state when
            the object is called.
        relaxation_timescale : DataArray, optional
            The timescale tau with which the Newtonian relaxation occurs.
            If not given, it should be provided in the state when
            the object is called.
        """
        self._quantity_name = quantity_name
        self._equilibrium_value = equilibrium_value
        self._tau = relaxation_timescale

    def __call__(self, state):
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Below, (quantity_name)
            refers to the quantity_name passed at initialization. The state
            must contain:

            * (quantity_name)
            * equilibrium_(quantity_name), unless this was passed at
              initialisation time in which case that value is used
            * (quantity_name)_relaxation_timescale, unless this was passed
              at initialisation time in which case that value is used

        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state.
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.
        """
        value = state[self._quantity_name]
        if self._equilibrium_value is None:
            equilibrium = state['equilibrium_' + self._quantity_name].to_units(
                value.attrs['units'])
        else:
            equilibrium = self._equilibrium_value.to_units(
                value.attrs['units'])
        if self._tau is None:
            tau = state[
                self._quantity_name + '_relaxation_timescale'].to_units(
                's')
        else:
            tau = self._tau.to_units('s')
        tendency_unit_string = str(
            ureg(state[self._quantity_name].attrs['units']) / ureg('s'))
        tendencies = {
            self._quantity_name: DataArray(
                (equilibrium.values - value.values)/tau.values,
                dims=value.dims,
                attrs={'units': tendency_unit_string}
            )
        }
        return tendencies, {}


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
