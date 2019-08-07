from .._core.base_components import (
    TendencyComponent, DiagnosticComponent, ImplicitTendencyComponent, Stepper
)


class ScalingWrapper(object):
    """
    Wraps any component and scales either inputs, outputs or tendencies
    by a floating point value.

    Example
    -------
    This is how the ScalingWrapper can be used to wrap a TendencyComponent.
    >>> scaled_component = ScalingWrapper(
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
        component : TendencyComponent, Stepper, DiagnosticComponent, ImplicitTendencyComponent
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
            The component is not of type Stepper or TendencyComponent.
        ValueError
            The keys in the scale factors do not correspond to valid
            input/output/tendency for this component.
        """
        if not any(
                isinstance(component, t) for t in [
                    DiagnosticComponent, TendencyComponent, ImplicitTendencyComponent, Stepper]):
            raise TypeError(
                'component must be a component type (DiagnosticComponent, TendencyComponent, '
                'ImplicitTendencyComponent, or Stepper)'
            )

        self._component = component
        self._input_scale_factors = dict()
        if input_scale_factors is not None:

            for input_field in input_scale_factors.keys():
                if input_field not in component.input_properties.keys():
                    raise ValueError(
                        "{} is not a valid input quantity.".format(input_field))

            self._input_scale_factors = input_scale_factors

        self._diagnostic_scale_factors = dict()
        if diagnostic_scale_factors is not None:
            if not hasattr(component, 'diagnostic_properties'):
                raise TypeError(
                    'Cannot apply diagnostic scale factors to component without '
                    'diagnostic output.')
            self._ensure_fields_have_properties(
                diagnostic_scale_factors, component.diagnostic_properties, 'diagnostic')
            self._diagnostic_scale_factors = diagnostic_scale_factors

        self._output_scale_factors = dict()
        if output_scale_factors is not None:
            if not hasattr(component, 'output_properties'):
                raise TypeError(
                    'Cannot apply output scale factors to component without '
                    'output_properties.')
            self._ensure_fields_have_properties(
                output_scale_factors, component.output_properties, 'output')
            self._output_scale_factors = output_scale_factors

        self._tendency_scale_factors = dict()
        if tendency_scale_factors is not None:
            if not hasattr(component, 'tendency_properties'):
                raise TypeError(
                    'Cannot apply tendency scale factors to component that does '
                    'not output tendencies.')
            self._ensure_fields_have_properties(
                tendency_scale_factors, component.tendency_properties, 'tendency')
            self._tendency_scale_factors = tendency_scale_factors

    def _ensure_fields_have_properties(
            self, scale_factors, properties, properties_name):
        for field in scale_factors.keys():
            if field not in properties.keys():
                raise ValueError(
                    "{} is not a {} quantity in the given component"
                    ", but was given a scale factor.".format(field, properties_name))

    def __getattr__(self, item):
        return getattr(self._component, item)

    def __call__(self, state, timestep=None):
        """
        Call the underlying component, applying scaling.

        Parameters
        ----------
        state : dict
            A model state dictionary.
        timestep : timedelta, optional
            A time step. If the underlying component does not use a timestep,
            this will be discarded. If it does, this argument is required.

        Returns
        -------
        *args
            The return values of the underlying component.
        """
        scaled_state = {}
        if 'time' in state:
            scaled_state['time'] = state['time']

        for input_field in self.input_properties.keys():
            if input_field in self._input_scale_factors:
                scale_factor = self._input_scale_factors[input_field]
                scaled_state[input_field] = state[input_field]*float(scale_factor)
                scaled_state[input_field].attrs = state[input_field].attrs
            else:
                scaled_state[input_field] = state[input_field]

        if isinstance(self._component, Stepper):
            if timestep is None:
                raise TypeError('Must give timestep to call Stepper.')
            diagnostics, new_state = self._component(scaled_state, timestep)
            for name in self._output_scale_factors.keys():
                scale_factor = self._output_scale_factors[name]
                new_state[name] *= float(scale_factor)
            for name in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[name]
                diagnostics[name] *= float(scale_factor)
            return diagnostics, new_state
        elif isinstance(self._component, TendencyComponent):
            tendencies, diagnostics = self._component(scaled_state)
            for tend_field in self._tendency_scale_factors.keys():
                scale_factor = self._tendency_scale_factors[tend_field]
                tendencies[tend_field] *= float(scale_factor)
            for name in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[name]
                diagnostics[name] *= float(scale_factor)
            return tendencies, diagnostics
        elif isinstance(self._component, ImplicitTendencyComponent):
            if timestep is None:
                raise TypeError('Must give timestep to call ImplicitTendencyComponent.')
            tendencies, diagnostics = self._component(scaled_state, timestep)
            for tend_field in self._tendency_scale_factors.keys():
                scale_factor = self._tendency_scale_factors[tend_field]
                tendencies[tend_field] *= float(scale_factor)
            for name in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[name]
                diagnostics[name] *= float(scale_factor)
            return tendencies, diagnostics
        elif isinstance(self._component, DiagnosticComponent):
            diagnostics = self._component(scaled_state)
            for name in self._diagnostic_scale_factors.keys():
                scale_factor = self._diagnostic_scale_factors[name]
                diagnostics[name] *= float(scale_factor)
            return diagnostics
        else:  # Should never reach this
            raise RuntimeError(
                'Unknown component type, seems to be a bug in ScalingWrapper')


class UpdateFrequencyWrapper(object):
    """
    Wraps a component object so that when it is called, it only computes new
    output if sufficient time has passed, and otherwise returns its last
    computed output.

    Example
    -------
    This how the wrapper should be used on a fictional TendencyComponent class
    called MyPrognostic.
    >>> from datetime import timedelta
    >>> prognostic = UpdateFrequencyWrapper(MyPrognostic(), timedelta(hours=1))
    """

    def __init__(self, component, update_timedelta):
        """
        Initialize the UpdateFrequencyWrapper object.

        Args
        ----
        component : TendencyComponent, Stepper, DiagnosticComponent, ImplicitTendencyComponent
            The component to be wrapped.
        update_timedelta : timedelta
            The amount that state['time'] must differ from when output
            was cached before new output is computed.
        """
        self.component = component
        self._update_timedelta = update_timedelta
        self._cached_output = None
        self._last_update_time = None

    def __call__(self, state, timestep=None, **kwargs):
        """
        Call the underlying component, or return cached values instead if
        insufficient time has passed since the last time cached values were
        stored.

        Parameters
        ----------
        state : dict
            A model state dictionary.
        timestep : timedelta, optional
            A time step. If the underlying component does not use a timestep,
            this will be discarded. If it does, this argument is required.

        Returns
        -------
        *args
            The return values of the underlying component.
        """
        if ((self._last_update_time is None) or
                (state['time'] >= self._last_update_time +
                 self._update_timedelta)):
            if timestep is not None:
                try:
                    self._cached_output = self.component(state, timestep, **kwargs)
                except TypeError:
                    self._cached_output = self.component(state, **kwargs)
            else:
                self._cached_output = self.component(state, **kwargs)
            self._last_update_time = state['time']
        return self._cached_output

    def __getattr__(self, item):
        return getattr(self.component, item)
