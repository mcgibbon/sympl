from .._core.base_components import (
    Prognostic, Diagnostic, ImplicitPrognostic, Implicit
)


class InputScalingMixin(object):

    @property
    def input_properties(self):
        return self.wrapped_component.input_properties

    def __init__(self, input_scale_factors=None):
        self.input_scale_factors = dict()
        if input_scale_factors is not None:
            for input_field, value in input_scale_factors.items():
                if input_field not in self.wrapped_component.inputs:
                    raise ValueError(
                        "{} is not an input of the wrapped component.".format(input_field))
                self.input_scale_factors[input_field] = value
        super(InputScalingMixin, self).__init__()

    def apply_input_scaling(self, input_state):
        scaled_state = {}
        for name in scaled_state.keys():
            if name in self.input_scale_factors.keys():
                scale_factor = self.input_scale_factors[name]
                scaled_state[name] = input_state[name]*scale_factor
                scaled_state[name].attrs.update(input_state[name].attrs)
            else:
                scaled_state[name] = input_state[name]
        return scaled_state


class OutputScalingMixin(object):

    @property
    def output_properties(self):
        return self.wrapped_component.output_properties

    def __init__(self, output_scale_factors=None):
        self.output_scale_factors = dict()
        if output_scale_factors is not None:
            for input_field, value in output_scale_factors.items():
                if input_field not in self.wrapped_component.inputs:
                    raise ValueError(
                        "{} is not an input of the wrapped component.".format(
                            input_field))
                self.output_scale_factors[input_field] = value
        super(OutputScalingMixin, self).__init__()

    def apply_output_scaling(self, output_state):
        scaled_outputs = {}
        for name in scaled_outputs.keys():
            if name in self.output_scale_factors.keys():
                scale_factor = self.output_scale_factors[name]
                scaled_outputs[name] = output_state[name] * scale_factor
                scaled_outputs[name].attrs.update(output_state[name].attrs)
            else:
                scaled_outputs[name] = output_state[name]
        return scaled_outputs


class DiagnosticScalingMixin(object):

    @property
    def diagnostic_properties(self):
        return self.wrapped_component.diagnostic_properties

    def __init__(self, diagnostic_scale_factors=None):
        self.diagnostic_scale_factors = dict()
        if diagnostic_scale_factors is not None:
            for input_field, value in diagnostic_scale_factors.items():
                if input_field not in self.wrapped_component.inputs:
                    raise ValueError(
                        "{} is not an input of the wrapped component.".format(
                            input_field))
                self.diagnostic_scale_factors[input_field] = value
        super(DiagnosticScalingMixin, self).__init__()

    def apply_diagnostic_scaling(self, diagnostics):
        scaled_diagnostics = {}
        for name in scaled_diagnostics.keys():
            if name in self.diagnostic_scale_factors.keys():
                scale_factor = self.diagnostic_scale_factors[name]
                scaled_diagnostics[name] = diagnostics[name] * scale_factor
                scaled_diagnostics[name].attrs.update(diagnostics[name].attrs)
            else:
                scaled_diagnostics[name] = diagnostics[name]
        return scaled_diagnostics


class TendencyScalingMixin(object):

    @property
    def tendency_properties(self):
        return self.wrapped_component.tendency_properties

    def __init__(self, tendency_scale_factors=None):
        self.tendency_scale_factors = dict()
        if tendency_scale_factors is not None:
            for input_field, value in tendency_scale_factors.items():
                if input_field not in self.wrapped_component.inputs:
                    raise ValueError(
                        "{} is not an input of the wrapped component.".format(
                            input_field))
                self.tendency_scale_factors[input_field] = value
        super(TendencyScalingMixin, self).__init__()

    def apply_tendency_scaling(self, tendencies):
        scaled_tendencies = {}
        for name in scaled_tendencies.keys():
            if name in self.tendency_scale_factors.keys():
                scale_factor = self.tendency_scale_factors[name]
                scaled_tendencies[name] = tendencies[name] * scale_factor
                scaled_tendencies[name].attrs.update(tendencies[name].attrs)
            else:
                scaled_tendencies[name] = tendencies[name]
        return scaled_tendencies


class ScalingWrapper(object):

    def __init__(self, component):
        """
        Initializes the scaling wrapper.

        Parameters
        ----------
        component
            The component to be wrapped.
        """
        self.wrapped_component = component
        super(ScalingWrapper, self).__init__()


class PrognosticScalingWrapper(
    ScalingWrapper, Prognostic, InputScalingMixin, DiagnosticScalingMixin, TendencyScalingMixin):

    def __call__(self, state):
        input = self.apply_input_scaling(state)
        tendencies, diagnostics = self.wrapped_component(input)
        tendencies = self.apply_tendency_scaling(tendencies)
        diagnostics = self.apply_diagnostic_scaling(diagnostics)
        return tendencies, diagnostics


class ImplicitPrognosticScalingWrapper(
    ScalingWrapper, ImplicitPrognostic, InputScalingMixin, DiagnosticScalingMixin, TendencyScalingMixin):

    def __call__(self, state, timestep):
        input = self.apply_input_scaling(state)
        tendencies, diagnostics = self.wrapped_component(input, timestep)
        tendencies = self.apply_tendency_scaling(tendencies)
        diagnostics = self.apply_diagnostic_scaling(diagnostics)
        return tendencies, diagnostics


class DiagnosticScalingWrapper(
    ScalingWrapper, Diagnostic, InputScalingMixin, DiagnosticScalingMixin):

    def __call__(self, state):
        input = self.apply_input_scaling(state)
        tendencies, diagnostics = self.wrapped_component(input)
        tendencies = self.apply_tendency_scaling(tendencies)
        diagnostics = self.apply_diagnostic_scaling(diagnostics)
        return tendencies, diagnostics


class ImplicitScalingWrapper(
    ScalingWrapper, Diagnostic, InputScalingMixin, DiagnosticScalingMixin, OutputScalingMixin):

    def __call__(self, state, timestep):
        input = self.apply_input_scaling(state)
        diagnostics, output = self.wrapped_component(input, timestep)
        diagnostics = self.apply_diagnostic_scaling(diagnostics)
        output = self.apply_output_scaling(output)
        return diagnostics, output


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
