from .base_components import ImplicitPrognostic
from .array import DataArray


class ScalingWrapper(object):
    """
    Wraps any component and scales either inputs, outputs or tendencies
    by a floating point value.
    Example
    -------
    This is how the ScaledInputOutputWrapper can be used to wrap a Prognostic.
    >>> scaled_component = ScaledInputOutputWrapper(
    >>>     RRTMRadiation(),
    >>>     input_scale_factors = {
    >>>         'specific_humidity' = 0.2},
    >>>     tendency_scale_factors = {
    >>>         'air_temperature' = 1.5})
    """

    def __init__(self,
                 component,
                 input_scale_factors=None,
                 output_scale_factors=None,
                 tendency_scale_factors=None,
                 diagnostic_scale_factors=None):
        """
        Initializes the ScaledInputOutputWrapper object.
        Args
        ----
        component : Prognostic, Implicit
            The component to be wrapped.
        input_scale_factors : dict
            a dictionary whose keys are the inputs that will be scaled
            and values are floating point scaling factors.
        output_scale_factors : dict
            a dictionary whose keys are the outputs that will be scaled
            and values are floating point scaling factors.
        tendency_scale_factors : dict
            a dictionary whose keys are the tendencies that will be scaled
            and values are floating point scaling factors.
        diagnostic_scale_factors : dict
            a dictionary whose keys are the diagnostics that will be scaled
            and values are floating point scaling factors.
        Returns
        -------
        scaled_component : ScaledInputOutputWrapper
            the scaled version of the component
        Raises
        ------
        TypeError
            The component is not of type Implicit or Prognostic.
        ValueError
            The keys in the scale factors do not correspond to valid
            input/output/tendency for this component.
        """

        self._input_scale_factors = dict()
        if input_scale_factors is not None:

            for input_field in input_scale_factors.keys():
                if input_field not in component.inputs:
                    raise ValueError(
                        "{} is not a valid input quantity.".format(input_field))

            self._input_scale_factors = input_scale_factors

        self._diagnostic_scale_factors = dict()
        if diagnostic_scale_factors is not None:

            for diagnostic_field in diagnostic_scale_factors.keys():
                if diagnostic_field not in component.diagnostics:
                    raise ValueError(
                        "{} is not a valid diagnostic quantity.".format(diagnostic_field))

            self._diagnostic_scale_factors = diagnostic_scale_factors

        if hasattr(component, 'input_properties') and hasattr(component, 'output_properties'):

            self._output_scale_factors = dict()
            if output_scale_factors is not None:

                for output_field in output_scale_factors.keys():
                    if output_field not in component.outputs:
                        raise ValueError(
                            "{} is not a valid output quantity.".format(output_field))

                self._output_scale_factors = output_scale_factors
            self._component_type = 'Implicit'

        elif hasattr(component, 'input_properties') and hasattr(component, 'tendency_properties'):

            self._tendency_scale_factors = dict()
            if tendency_scale_factors is not None:

                for tendency_field in tendency_scale_factors.keys():
                    if tendency_field not in component.tendencies:
                        raise ValueError(
                            "{} is not a valid tendency quantity.".format(tendency_field))

                self._tendency_scale_factors = tendency_scale_factors
            self._component_type = 'Prognostic'

        elif hasattr(component, 'input_properties') and hasattr(component, 'diagnostic_properties'):
            self._component_type = 'Diagnostic'
        else:
            raise TypeError(
                "Component must be either of type Implicit or Prognostic or Diagnostic")

        self._component = component

    def __getattr__(self, item):
        return getattr(self._component, item)

    def __call__(self, state, timestep=None):

        scaled_state = {}
        if 'time' in state:
            scaled_state['time'] = state['time']

        for input_field in self.inputs:
            if input_field in self._input_scale_factors:
                scale_factor = self._input_scale_factors[input_field]
                scaled_state[input_field] = state[input_field]*float(scale_factor)
            else:
                scaled_state[input_field] = state[input_field]

        if self._component_type == 'Implicit':
            diagnostics, new_state = self._component(scaled_state, timestep)

            for output_field in self._output_scale_factors.keys():
                scale_factor = self._output_scale_factors[output_field]
                new_state[output_field] *= float(scale_factor)

            for diagnostic_field in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[diagnostic_field]
                diagnostics[diagnostic_field] *= float(scale_factor)

            return diagnostics, new_state
        elif self._component_type == 'Prognostic':
            tendencies, diagnostics = self._component(scaled_state)

            for tend_field in self._tendency_scale_factors.keys():
                scale_factor = self._tendency_scale_factors[tend_field]
                tendencies[tend_field] *= float(scale_factor)

            for diagnostic_field in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[diagnostic_field]
                diagnostics[diagnostic_field] *= float(scale_factor)

            return tendencies, diagnostics
        elif self._component_type == 'Diagnostic':
            diagnostics = self._component(scaled_state)

            for diagnostic_field in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[diagnostic_field]
                diagnostics[diagnostic_field] *= float(scale_factor)

            return diagnostics
        else:  # Should never reach this
            raise ValueError(
                'Unknown component type, seems to be a bug in ScalingWrapper')


class UpdateFrequencyWrapper(object):
    """
    Wraps a prognostic object so that when it is called, it only computes new
    output if sufficient time has passed, and otherwise returns its last
    computed output. The Delayed object requires that the 'time' attribute is
    set in the state, in addition to any requirements of the Prognostic

    Example
    -------
    This how the wrapper should be used on a fictional Prognostic class
    called MyPrognostic.

    >>> from datetime import timedelta
    >>> prognostic = UpdateFrequencyWrapper(MyPrognostic(), timedelta(hours=1))
    """

    def __init__(self, prognostic, update_timedelta):
        """
        Initialize the UpdateFrequencyWrapper object.

        Args
        ----
        prognostic : Prognostic
            The object to be wrapped.
        update_timedelta : timedelta
            The amount that state['time'] must differ from when output
            was cached before new output is computed.
        """
        self._prognostic = prognostic
        self._update_timedelta = update_timedelta
        self._cached_output = None
        self._last_update_time = None

    def __call__(self, state, **kwargs):
        if ((self._last_update_time is None) or
                (state['time'] >= self._last_update_time +
                 self._update_timedelta)):
            self._cached_output = self._prognostic(state, **kwargs)
            self._last_update_time = state['time']
        return self._cached_output

    def __getattr__(self, item):
        return getattr(self._prognostic, item)


class TendencyInDiagnosticsWrapper(object):
    """
    Wraps a prognostic object so that when it is called, it returns all
    tendencies in its diagnostics.

    Example
    -------
    This how the wrapper should be used on a fictional Prognostic class
    called RRTMRadiation.

    >>> prognostic = TendencyInDiagnosticsWrapper(RRTMRadiation(), 'radiation')
    """

    def __init__(self, prognostic, label):
        """
        Initialize the Delayed object.

        Args
        ----
        prognostic : Prognostic
            The object to be wrapped
        label : str
            Label describing what the tendencies are due to, to be
            put in the diagnostic quantity names.
        """
        self._prognostic = prognostic
        self._tendency_label = label
        self._tendency_diagnostic_properties = {}
        for quantity_name, properties in prognostic.tendency_properties.items():
            diagnostic_name = 'tendency_of_{}_due_to_{}'.format(quantity_name, label)
            self._tendency_diagnostic_properties[diagnostic_name] = properties

    @property
    def inputs(self):
        return list(self.diagnostic_properties.keys())

    @property
    def input_properties(self):
        return_dict = self._prognostic.input_properties.copy()
        return_dict.update(self._tendency_diagnostic_properties)
        return return_dict

    @property
    def diagnostics(self):
        return list(self.diagnostic_properties.keys())

    @property
    def diagnostic_properties(self):
        return_dict = self._prognostic.diagnostic_properties.copy()
        return_dict.update(self._tendency_diagnostic_properties)
        return return_dict

    def __call__(self, state, **kwargs):
        tendencies, diagnostics = self._prognostic(state, **kwargs)
        for quantity_name in tendencies.keys():
            diagnostic_name = 'tendency_of_{}_due_to_{}'.format(
                quantity_name, self._tendency_label)
            diagnostics[diagnostic_name] = tendencies[quantity_name]
        return tendencies, diagnostics

    def __getattr__(self, item):
        return getattr(self._prognostic, item)


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
