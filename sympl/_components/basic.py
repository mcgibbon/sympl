from .._core.array import DataArray
from .._core.base_components import ImplicitPrognostic, Prognostic, Diagnostic
from .._core.units import unit_registry as ureg
from .._core.util import combine_dims


class ConstantPrognostic(Prognostic):
    """
    Prescribes constant tendencies provided at initialization.

    Note: Any arrays in the passed dictionaries are not copied, so that
        if you were to modify them after passing them into this object,
        it would also modify the values inside this object.
    """

    @property
    def input_properties(self):
        return {}

    @property
    def tendency_properties(self):
        return_dict = {}
        for name, data_array in self.__tendencies.items():
            return_dict[name] = {
                'dims': data_array.dims,
                'units': data_array.attrs['units'],
            }
        return return_dict

    @property
    def diagnostic_properties(self):
        return_dict = {}
        for name, data_array in self.__diagnostics.items():
            return_dict[name] = {
                'dims': data_array.dims,
                'units': data_array.attrs['units'],
            }
        return return_dict

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
        self.__tendencies = tendencies.copy()
        if diagnostics is not None:
            self.__diagnostics = diagnostics.copy()
        else:
            self.__diagnostics = {}
        super(ConstantPrognostic, self).__init__()

    def array_call(self, state):
        tendencies = {}
        for name, data_array in self.__tendencies.items():
            tendencies[name] = data_array.values
        diagnostics = {}
        for name, data_array in self.__diagnostics.items():
            diagnostics[name] = data_array.values
        return tendencies, diagnostics


class ConstantDiagnostic(Diagnostic):
    """
    Yields constant diagnostics provided at initialization.

    Note
    ----
    Any arrays in the passed dictionaries are not copied, so that
    if you were to modify them after passing them into this object,
    it would also modify the values inside this object.
    """

    @property
    def input_properties(self):
        return {}

    @property
    def diagnostic_properties(self):
        return_dict = {}
        for name, data_array in self.__diagnostics.items():
            return_dict[name] = {
                'dims': data_array.dims,
                'units': data_array.attrs['units'],
            }
        return return_dict

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
        self.__diagnostics = diagnostics.copy()
        super(ConstantDiagnostic, self).__init__()

    def array_call(self, state):
        return_state = {}
        for name, data_array in self.__diagnostics.items():
            return_state[name] = data_array.values
        return return_state


class RelaxationPrognostic(Prognostic):
    r"""
    Applies Newtonian relaxation to a single quantity.

    The relaxation takes the form
    :math:`\frac{dx}{dt} = - \frac{x - x_{eq}}{\tau}`
    where :math:`x` is the quantity being relaxed, :math:`x_{eq}` is the
    equilibrium value, and :math:`\tau` is the timescale of the relaxation.
    """

    @property
    def input_properties(self):
        return_dict = {
            self._quantity_name: {
                'dims': ['*'],
                'units': self._units,
            },
            'equilibrium_{}'.format(self._quantity_name): {
                'dims': ['*'],
                'units': self._units,
            },
            '{}_relaxation_timescale'.format(self._quantity_name): {
                'dims': ['*'],
                'units': 's',
            }
        }
        return return_dict

    @property
    def tendency_properties(self):
        return {
            self._quantity_name: {
                'dims': ['*'],
                'units': str(ureg(self._units) / ureg('s')),
            }
        }

    @property
    def diagnostic_properties(self):
        return {}

    def __init__(self, quantity_name, units, **kwargs):
        """
        Args
        ----
        quantity_name : str
            The name of the quantity to which Newtonian relaxation should be
            applied.
        units : str
            The units of the relaxed quantity as to be used internally when
            computing tendency. Can be any units convertible from the actual
            input you plan to use.
        input_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which input values are scaled before being used
            by this object.
        tendency_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which tendency values are scaled before being
            returned by this object.
        diagnostic_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which diagnostic values are scaled before being
            returned by this object.
        update_interval : timedelta, optional
            If given, the component will only give new output if at least
            a period of update_interval has passed since the last time new
            output was given. Otherwise, it would return that cached output.
        """
        self._quantity_name = quantity_name
        self._units = units
        super(RelaxationPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary as numpy arrays. Below, (quantity_name)
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
            quantities in units/second at the time of the input state, as
            numpy arrays.
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state, as numpy arrays.
        """
        value = state[self._quantity_name]
        equilibrium = state['equilibrium_' + self._quantity_name]
        tau = state[self._quantity_name + '_relaxation_timescale']
        tendencies = {
            self._quantity_name: (equilibrium - value)/tau
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

    def __init__(self, implicit, **kwargs):
        if len(kwargs) > 0:
            raise TypeError('Received unexpected keyword argument {}'.format(
                kwargs.popitem()[0]))
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

    def array_call(self, state, timestep):
        raise NotImplementedError()

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
