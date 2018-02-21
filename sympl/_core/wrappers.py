from .base_components import ImplicitPrognostic
from .array import DataArray


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


class ImplicitPrognosticWrapper(ImplicitPrognostic):

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
