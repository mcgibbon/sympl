from .._core.dataarray import DataArray
from .._core.base_components import ImplicitTendencyComponent, TendencyComponent, DiagnosticComponent
from .._core.units import unit_registry as ureg


class ConstantTendencyComponent(TendencyComponent):
    """
    Prescribes constant tendencies provided at initialization.

    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendency_properties : dict
        A dictionary whose keys are quantities for which tendencies are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    input_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which input values are scaled before being used
        by this object.
    tendency_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which tendency values are scaled before being
        returned by this object.
    diagnostic_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which diagnostic values are scaled before being
        returned by this object.
    update_interval : timedelta
        If not None, the component will only give new output if at least
        a period of update_interval has passed since the last time new
        output was given. Otherwise, it would return that cached output.
    tendencies_in_diagnostics : boo
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".

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

    def __init__(self, tendencies, diagnostics=None, **kwargs):
        """
        Args
        ----
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second to be returned by this TendencyComponent.
        diagnostics : dict, optional
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            to be returned by this TendencyComponent. By default an empty dictionary
            is used.
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
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        self.__tendencies = tendencies.copy()
        if diagnostics is not None:
            self.__diagnostics = diagnostics.copy()
        else:
            self.__diagnostics = {}
        super(ConstantTendencyComponent, self).__init__(**kwargs)

    def array_call(self, state):
        tendencies = {}
        for name, data_array in self.__tendencies.items():
            tendencies[name] = data_array.values
        diagnostics = {}
        for name, data_array in self.__diagnostics.items():
            diagnostics[name] = data_array.values
        return tendencies, diagnostics


class ConstantDiagnosticComponent(DiagnosticComponent):
    """
    Yields constant diagnostics provided at initialization.

    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    input_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which input values are scaled before being used
        by this object.
    diagnostic_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which diagnostic values are scaled before being
        returned by this object.
    update_interval : timedelta
        If not None, the component will only give new output if at least
        a period of update_interval has passed since the last time new
        output was given. Otherwise, it would return that cached output.

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

    def __init__(self, diagnostics, **kwargs):
        """
        Args
        ----
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities.
            The values in the dictionary will be returned when this
            DiagnosticComponent is called.
        input_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which input values are scaled before being used
            by this object.
        diagnostic_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which diagnostic values are scaled before being
            returned by this object.
        update_interval : timedelta, optional
            If given, the component will only give new output if at least
            a period of update_interval has passed since the last time new
            output was given. Otherwise, it would return that cached output.
        """
        self.__diagnostics = diagnostics.copy()
        super(ConstantDiagnosticComponent, self).__init__(**kwargs)

    def array_call(self, state):
        return_state = {}
        for name, data_array in self.__diagnostics.items():
            return_state[name] = data_array.values
        return return_state


class RelaxationTendencyComponent(TendencyComponent):
    r"""
    Applies Newtonian relaxation to a single quantity.

    The relaxation takes the form
    :math:`\frac{dx}{dt} = - \frac{x - x_{eq}}{\tau}`
    where :math:`x` is the quantity being relaxed, :math:`x_{eq}` is the
    equilibrium value, and :math:`\tau` is the timescale of the relaxation.

    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendency_properties : dict
        A dictionary whose keys are quantities for which tendencies are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    input_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which input values are scaled before being used
        by this object.
    tendency_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which tendency values are scaled before being
        returned by this object.
    diagnostic_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which diagnostic values are scaled before being
        returned by this object.
    update_interval : timedelta
        If not None, the component will only give new output if at least
        a period of update_interval has passed since the last time new
        output was given. Otherwise, it would return that cached output.
    tendencies_in_diagnostics : boo
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
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
        super(RelaxationTendencyComponent, self).__init__(**kwargs)

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


class TimeDifferencingWrapper(ImplicitTendencyComponent):
    """
    Wraps an Stepper object and turns it into an ImplicitTendencyComponent by applying
    simple first-order time differencing to determine tendencies.

    Example
    -------
    This how the wrapper should be used on an Stepper class
    called GridScaleCondensation.

    >>> component = TimeDifferencingWrapper(GridScaleCondensation())
    """

    @property
    def input_properties(self):
        return self._implicit.input_properties

    @property
    def tendency_properties(self):
        return_dict = {}
        for name, properties in self._implicit.output_properties.items():
            return_dict[name] = properties.copy()
            return_dict[name]['units'] += ' s^-1'
        return return_dict

    @property
    def diagnostic_properties(self):
        return self._implicit.diagnostic_properties

    def __init__(self, implicit, **kwargs):
        """
        Initializes the TimeDifferencingWrapper. Some kwargs of Stepper
        objects are not implemented, and should be applied instead on the
        Stepper object which is wrapped by this one.

        Parameters
        ----------
        implicit: Stepper
            An Stepper component to wrap.
        """
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

    def __getattr__(self, item):
        if item in ('outputs', 'output_properties'):
            raise AttributeError()
        else:
            return getattr(self._implicit, item)
