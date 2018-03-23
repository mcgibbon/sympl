import abc
from .util import (
    get_numpy_arrays_with_properties, restore_data_arrays_with_properties)
from .time import timedelta
from .exceptions import (
    InvalidPropertyDictError, ComponentExtraOutputError,
    ComponentMissingOutputError)


def apply_scale_factors(array_state, scale_factors):
    for key, factor in scale_factors.items():
        array_state[key] *= factor


class TendencyMixin(object):

    @property
    def _wanted_tendency_aliases(self):
        wanted_tendency_aliases = {}
        for name, properties in self.tendency_properties.items():
            wanted_tendency_aliases[name] = []
            if 'alias' in properties.keys():
                wanted_tendency_aliases[name].append(properties['alias'])
            if (name in self.input_properties.keys() and
                    'alias' in self.input_properties[name].keys()):
                wanted_tendency_aliases[name].append(self.input_properties[name]['alias'])
        return wanted_tendency_aliases

    def _check_missing_tendencies(self, tendency_dict):
        missing_tendencies = set()
        for name, aliases in self._wanted_tendency_aliases.items():
            if (name not in tendency_dict.keys() and
                    not any(alias in tendency_dict.keys() for alias in aliases)):
                missing_tendencies.add(name)
        if len(missing_tendencies) > 0:
            raise ComponentMissingOutputError(
                'Component {} did not compute tendencies for {}'.format(
                    self.__class__.__name__, ', '.join(missing_tendencies)))

    def _check_extra_tendencies(self, tendency_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_tendency_aliases.keys())
        for value_list in self._wanted_tendency_aliases.values():
            wanted_set.update(value_list)
        extra_tendencies = set(tendency_dict.keys()).difference(wanted_set)
        if len(extra_tendencies) > 0:
            raise ComponentExtraOutputError(
                'Component {} computed tendencies for {} which are not in '
                'tendency_properties'.format(
                    self.__class__.__name__, ', '.join(extra_tendencies)))

    def _check_tendencies(self, tendency_dict):
        self._check_missing_tendencies(tendency_dict)
        self._check_extra_tendencies(tendency_dict)


class DiagnosticMixin(object):

    @property
    def _wanted_diagnostic_aliases(self):
        wanted_diagnostic_aliases = {}
        for name, properties in self.diagnostic_properties.items():
            wanted_diagnostic_aliases[name] = []
            if 'alias' in properties.keys():
                wanted_diagnostic_aliases[name].append(properties['alias'])
            if (name in self.input_properties.keys() and
                    'alias' in self.input_properties[name].keys()):
                wanted_diagnostic_aliases[name].append(self.input_properties[name]['alias'])
        return wanted_diagnostic_aliases

    def _check_missing_diagnostics(self, diagnostics_dict):
        missing_diagnostics = set()
        for name, aliases in self._wanted_diagnostic_aliases.items():
            if (name not in diagnostics_dict.keys() and
                    not any(alias in diagnostics_dict.keys() for alias in aliases)):
                missing_diagnostics.add(name)
        if len(missing_diagnostics) > 0:
            raise ComponentMissingOutputError(
                'Component {} did not compute diagnostics {}'.format(
                    self.__class__.__name__, ', '.join(missing_diagnostics)))

    def _check_extra_diagnostics(self, diagnostics_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_diagnostic_aliases.keys())
        for value_list in self._wanted_diagnostic_aliases.values():
            wanted_set.update(value_list)
        extra_diagnostics = set(diagnostics_dict.keys()).difference(wanted_set)
        if len(extra_diagnostics) > 0:
            raise ComponentExtraOutputError(
                'Component {} computed diagnostics {} which are not in '
                'diagnostic_properties'.format(
                    self.__class__.__name__, ', '.join(extra_diagnostics)))

    def _check_diagnostics(self, diagnostics_dict):
        self._check_missing_diagnostics(diagnostics_dict)
        self._check_extra_diagnostics(diagnostics_dict)


class OutputMixin(object):

    @property
    def _wanted_output_aliases(self):
        wanted_output_aliases = {}
        for name, properties in self.output_properties.items():
            wanted_output_aliases[name] = []
            if 'alias' in properties.keys():
                wanted_output_aliases[name].append(properties['alias'])
            if (name in self.input_properties.keys() and
                    'alias' in self.input_properties[name].keys()):
                wanted_output_aliases[name].append(
                    self.input_properties[name]['alias'])
        return wanted_output_aliases

    def _check_missing_outputs(self, outputs_dict):
        missing_outputs = set()
        for name, aliases in self._wanted_output_aliases.items():
            if (name not in outputs_dict.keys() and
                    not any(alias in outputs_dict.keys() for alias in
                            aliases)):
                missing_outputs.add(name)
        if len(missing_outputs) > 0:
            raise ComponentMissingOutputError(
                'Component {} did not compute outputs {}'.format(
                    self.__class__.__name__, ', '.join(missing_outputs)))

    def _check_extra_outputs(self, outputs_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_output_aliases.keys())
        for value_list in self._wanted_output_aliases.values():
            wanted_set.update(value_list)
        extra_outputs = set(outputs_dict.keys()).difference(wanted_set)
        if len(extra_outputs) > 0:
            raise ComponentExtraOutputError(
                'Component {} computed outputs {} which are not in '
                'output_properties'.format(
                    self.__class__.__name__, ', '.join(extra_outputs)))

    def _check_outputs(self, output_dict):
        self._check_missing_outputs(output_dict)
        self._check_extra_outputs(output_dict)


class Implicit(DiagnosticMixin, OutputMixin):
    """
    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are quantities for which values
        for the old state are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    output_properties : dict
        A dictionary whose keys are quantities for which values
        for the new state are returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    input_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which input values are scaled before being used
        by this object.
    output_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which output values are scaled before being
        returned by this object.
    diagnostic_scale_factors : dict
        A (possibly empty) dictionary whose keys are quantity names and
        values are floats by which diagnostic values are scaled before being
        returned by this object.
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    update_interval : timedelta
        If not None, the component will only give new output if at least
        a period of update_interval has passed since the last time new
        output was given. Otherwise, it would return that cached output.
    time_unit_name : str
        The unit to use for time differencing when putting tendencies in
        diagnostics.
    time_unit_timedelta: timedelta
        A timedelta corresponding to a single time unit as used for time
        differencing when putting tendencies in diagnostics.
    """
    __metaclass__ = abc.ABCMeta

    time_unit_name = 's'
    time_unit_timedelta = timedelta(seconds=1)

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    @abc.abstractproperty
    def output_properties(self):
        return {}

    def __str__(self):
        return (
            'instance of {}(Implicit)\n'
            '    inputs: {}\n'
            '    outputs: {}\n'
            '    diagnostics: {}'.format(
                self.__class__, self.input_properties.keys(),
                self.output_properties.keys(),
                self.diagnostic_properties.keys())
        )

    def __repr__(self):
        if hasattr(self, '_making_repr') and self._making_repr:
            return '{}(recursive reference)'.format(self.__class__)
        else:
            self._making_repr = True
            return_value = '{}({})'.format(
                self.__class__,
                '\n'.join('{}: {}'.format(repr(key), repr(value))
                          for key, value in self.__dict__.items()
                          if key != '_making_repr'))
            self._making_repr = False
            return return_value

    def __init__(
            self, input_scale_factors=None, output_scale_factors=None,
            diagnostic_scale_factors=None, tendencies_in_diagnostics=False,
            update_interval=None, name=None):
        """
        Initializes the Implicit object.

        Args
        ----
        input_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which input values are scaled before being used
            by this object.
        output_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which output values are scaled before being
            returned by this object.
        diagnostic_scale_factors : dict, optional
            A (possibly empty) dictionary whose keys are quantity names and
            values are floats by which diagnostic values are scaled before being
            returned by this object.
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output based on first order time
            differencing of output values.
        update_interval : timedelta, optional
            If given, the component will only give new output if at least
            a period of update_interval has passed since the last time new
            output was given. Otherwise, it would return that cached output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        if input_scale_factors is not None:
            self.input_scale_factors = input_scale_factors
        else:
            self.input_scale_factors = {}
        if output_scale_factors is not None:
            self.output_scale_factors = output_scale_factors
        else:
            self.output_scale_factors = {}
        if diagnostic_scale_factors is not None:
            self.diagnostic_scale_factors = diagnostic_scale_factors
        else:
            self.diagnostic_scale_factors = {}
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.update_interval = update_interval
        self._last_update_time = None
        if name is None:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name
        if tendencies_in_diagnostics:
            self._added_tendency_properties = self._insert_tendency_properties()
        else:
            self._added_tendency_properties = set()

    def _insert_tendency_properties(self):
        added_names = []
        for name, properties in self.output_properties.items():
            tendency_name = self._get_tendency_name(name)
            if properties['units'] is '':
                units = 's^-1'
            else:
                units = '{} s^-1'.format(properties['units'])
            self.diagnostic_properties[tendency_name] = {
                'units': units,
                'dims': properties['dims'],
            }
            if name not in self.input_properties.keys():
                self.input_properties[name] = {
                    'dims': properties['dims'],
                    'units': properties['units'],
                }
            elif self.input_properties[name]['dims'] != self.output_properties[name]['dims']:
                raise InvalidPropertyDictError(
                    'Can only calculate tendencies when input and output values'
                    ' have the same dimensions, but dims for {} are '
                    '{} (input) and {} (output)'.format(
                        name, self.input_properties[name]['dims'],
                        self.output_properties[name]['dims']
                    )
                )
            elif self.input_properties[name]['units'] != self.output_properties[name]['units']:
                raise InvalidPropertyDictError(
                    'Can only calculate tendencies when input and output values'
                    ' have the same units, but units for {} are '
                    '{} (input) and {} (output)'.format(
                        name, self.input_properties[name]['units'],
                        self.output_properties[name]['units']
                    )
                )
            added_names.append(tendency_name)
        return added_names

    def _check_missing_diagnostics(self, diagnostics_dict):
        missing_diagnostics = set()
        for name, aliases in self._wanted_diagnostic_aliases.items():
            if (name not in diagnostics_dict.keys() and
                    name not in self._added_tendency_properties and
                    not any(alias in diagnostics_dict.keys() for alias in aliases)):
                missing_diagnostics.add(name)
        if len(missing_diagnostics) > 0:
            raise ComponentMissingOutputError(
                'Component {} did not compute diagnostics {}'.format(
                    self.__class__.__name__, ', '.join(missing_diagnostics)))

    def _get_tendency_name(self, name):
        return '{}_tendency_from_{}'.format(name, self.name)

    @property
    def tendencies_in_diagnostics(self):
        return self._tendencies_in_diagnostics  # value cannot be modified

    def __call__(self, state, timestep):
        """
        Gets diagnostics from the current model state and steps the state
        forward in time according to the timestep.

        Args
        ----
        state : dict
            A model state dictionary.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the timestep after input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Implicit instance
            for other reasons.
        """
        if (self.update_interval is None or
                self._last_update_time is None or
                state['time'] >= self._last_update_time + self.update_interval):
            raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
            raw_state['time'] = state['time']
            apply_scale_factors(raw_state, self.input_scale_factors)
            raw_diagnostics, raw_new_state = self.array_call(raw_state, timestep)
            self._check_diagnostics(raw_diagnostics)
            self._check_outputs(raw_new_state)
            apply_scale_factors(raw_diagnostics, self.diagnostic_scale_factors)
            apply_scale_factors(raw_new_state, self.output_scale_factors)
            if self.tendencies_in_diagnostics:
                self._insert_tendencies_to_diagnostics(
                    raw_state, raw_new_state, timestep, raw_diagnostics)
            self._diagnostics = restore_data_arrays_with_properties(
                raw_diagnostics, self.diagnostic_properties,
                state, self.input_properties)
            self._new_state = restore_data_arrays_with_properties(
                raw_new_state, self.output_properties,
                state, self.input_properties)
            self._last_update_time = state['time']
        return self._diagnostics, self._new_state

    def _insert_tendencies_to_diagnostics(
            self, raw_state, raw_new_state, timestep, raw_diagnostics):
        for name in self.output_properties.keys():
            tendency_name = self._get_tendency_name(name)
            raw_diagnostics[tendency_name] = (
                (raw_new_state[name] - raw_state[name]) /
                timestep.total_seconds() * self.time_unit_timedelta.total_seconds())

    @abc.abstractmethod
    def array_call(self, state, timestep):
        """
        Gets diagnostics from the current model state and steps the state
        forward in time according to the timestep.

        Args
        ----
        state : dict
            A numpy array state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input properties of this
            object.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state, as numpy arrays.
        new_state : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the timestep after input state, as numpy arrays.
        """
        pass


class Prognostic(DiagnosticMixin, TendencyMixin):
    """
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
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def tendency_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    def __str__(self):
        return (
            'instance of {}(Prognostic)\n'
            '    inputs: {}\n'
            '    tendencies: {}\n'
            '    diagnostics: {}'.format(
                self.__class__, self.input_properties.keys(),
                self.tendency_properties.keys(),
                self.diagnostic_properties.keys())
        )

    def __repr__(self):
        if hasattr(self, '_making_repr') and self._making_repr:
            return '{}(recursive reference)'.format(self.__class__)
        else:
            self._making_repr = True
            return_value = '{}({})'.format(
                self.__class__,
                '\n'.join('{}: {}'.format(repr(key), repr(value))
                          for key, value in self.__dict__.items()
                          if key != '_making_repr'))
            self._making_repr = False
            return return_value

    def __init__(
            self, input_scale_factors=None, tendency_scale_factors=None,
            diagnostic_scale_factors=None, update_interval=None):
        """
        Initializes the Implicit object.

        Args
        ----
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
        if input_scale_factors is not None:
            self.input_scale_factors = input_scale_factors
        else:
            self.input_scale_factors = {}
        if tendency_scale_factors is not None:
            self.tendency_scale_factors = tendency_scale_factors
        else:
            self.tendency_scale_factors = {}
        if diagnostic_scale_factors is not None:
            self.diagnostic_scale_factors = diagnostic_scale_factors
        else:
            self.diagnostic_scale_factors = {}
        self.update_interval = update_interval
        self._last_update_time = None

    def __call__(self, state):
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary.

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

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Prognostic instance.
        """
        if (self.update_interval is None or
                self._last_update_time is None or
                state['time'] >= self._last_update_time + self.update_interval):
            raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
            raw_state['time'] = state['time']
            apply_scale_factors(raw_state, self.input_scale_factors)
            raw_tendencies, raw_diagnostics = self.array_call(raw_state)
            self._check_tendencies(raw_tendencies)
            self._check_diagnostics(raw_diagnostics)
            apply_scale_factors(raw_tendencies, self.tendency_scale_factors)
            apply_scale_factors(raw_diagnostics, self.diagnostic_scale_factors)
            self._tendencies = restore_data_arrays_with_properties(
                raw_tendencies, self.tendency_properties,
                state, self.input_properties)
            self._diagnostics = restore_data_arrays_with_properties(
                raw_diagnostics, self.diagnostic_properties,
                state, self.input_properties)
            self._last_update_time = state['time']
        return self._tendencies, self._diagnostics

    @abc.abstractmethod
    def array_call(self, state):
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input properties of this
            object.

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
        pass


class ImplicitPrognostic(DiagnosticMixin, TendencyMixin):
    """
    Attributes
    ----------
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    tendency_properties : dict
        A dictionary whose keys are quantities for which tendencies are returned
        when the object is called, and values are dictionaries which indicate
        'dims' and 'units'.
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
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y". By default the class name in
        lowercase is used.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def tendency_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    def __str__(self):
        return (
            'instance of {}(Prognostic)\n'
            '    inputs: {}\n'
            '    tendencies: {}\n'
            '    diagnostics: {}'.format(
                self.__class__, self.inputs, self.tendencies, self.diagnostics)
        )

    def __repr__(self):
        if hasattr(self, '_making_repr') and self._making_repr:
            return '{}(recursive reference)'.format(self.__class__)
        else:
            self._making_repr = True
            return_value = '{}({})'.format(
                self.__class__,
                '\n'.join('{}: {}'.format(repr(key), repr(value))
                          for key, value in self.__dict__.items()
                          if key != '_making_repr'))
            self._making_repr = False
            return return_value

    def __init__(
            self, input_scale_factors=None, tendency_scale_factors=None,
            diagnostic_scale_factors=None, update_interval=None, name=None):
        """
        Initializes the Implicit object.

        Args
        ----
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
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        update_interval : timedelta, optional
            If given, the component will only give new output if at least
            a period of update_interval has passed since the last time new
            output was given. Otherwise, it would return that cached output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        if input_scale_factors is not None:
            self.input_scale_factors = input_scale_factors
        else:
            self.input_scale_factors = {}
        if tendency_scale_factors is not None:
            self.tendency_scale_factors = tendency_scale_factors
        else:
            self.tendency_scale_factors = {}
        if diagnostic_scale_factors is not None:
            self.diagnostic_scale_factors = diagnostic_scale_factors
        else:
            self.diagnostic_scale_factors = {}
        self.update_interval = update_interval
        self._last_update_time = None
        if name is None:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name

    def __call__(self, state, timestep):
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary.
        timestep : timedelta
            The time over which the model is being stepped.

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

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Prognostic instance.
        """
        if (self.update_interval is None or
                self._last_update_time is None or
                state['time'] >= self._last_update_time + self.update_interval):
            raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
            raw_state['time'] = state['time']
            apply_scale_factors(raw_state, self.input_scale_factors)
            raw_tendencies, raw_diagnostics = self.array_call(raw_state, timestep)
            self._check_tendencies(raw_tendencies)
            self._check_diagnostics(raw_diagnostics)
            apply_scale_factors(raw_tendencies, self.tendency_scale_factors)
            apply_scale_factors(raw_diagnostics, self.diagnostic_scale_factors)
            self._tendencies = restore_data_arrays_with_properties(
                raw_tendencies, self.tendency_properties,
                state, self.input_properties)
            self._diagnostics = restore_data_arrays_with_properties(
                raw_diagnostics, self.diagnostic_properties,
                state, self.input_properties)
            self._last_update_time = state['time']
        return self._tendencies, self._diagnostics

    @abc.abstractmethod
    def array_call(self, state, timestep):
        """
        Gets tendencies and diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input properties of this
            object.
        timestep : timedelta
            The time over which the model is being stepped.

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


class Diagnostic(DiagnosticMixin):
    """
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
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    def __str__(self):
        return (
            'instance of {}(Diagnostic)\n'
            '    inputs: {}\n'
            '    diagnostics: {}'.format(
                self.__class__, self.inputs, self.diagnostics)
        )

    def __repr__(self):
        if hasattr(self, '_making_repr') and self._making_repr:
            return '{}(recursive reference)'.format(self.__class__)
        else:
            self._making_repr = True
            return_value = '{}({})'.format(
                self.__class__,
                '\n'.join('{}: {}'.format(repr(key), repr(value))
                          for key, value in self.__dict__.items()
                          if key != '_making_repr'))
            self._making_repr = False
            return return_value

    def __init__(
            self, input_scale_factors=None, diagnostic_scale_factors=None,
            update_interval=None):
        """
        Initializes the Implicit object.

        Args
        ----
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
        if input_scale_factors is not None:
            self.input_scale_factors = input_scale_factors
        else:
            self.input_scale_factors = {}
        if diagnostic_scale_factors is not None:
            self.diagnostic_scale_factors = diagnostic_scale_factors
        else:
            self.diagnostic_scale_factors = {}
        self.update_interval = update_interval
        self._last_update_time = None
        self._diagnostics = None

    def __call__(self, state):
        """
        Gets diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary.

        Returns
        -------
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Prognostic instance.
        """
        if (self.update_interval is None or
                self._last_update_time is None or
                state['time'] >= self._last_update_time + self.update_interval):
            raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
            raw_state['time'] = state['time']
            apply_scale_factors(raw_state, self.input_scale_factors)
            raw_diagnostics = self.array_call(raw_state)
            self._check_diagnostics(raw_diagnostics)
            apply_scale_factors(raw_diagnostics, self.diagnostic_scale_factors)
            self._diagnostics = restore_data_arrays_with_properties(
                raw_diagnostics, self.diagnostic_properties,
                state, self.input_properties)
            self._last_update_time = state['time']
        return self._diagnostics

    @abc.abstractmethod
    def array_call(self, state):
        """
        Gets diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary. Instead of data arrays, should
            include numpy arrays that satisfy the input properties of this
            object.

        Returns
        -------
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state, as numpy arrays.
        """


class Monitor(object):
    __metaclass__ = abc.ABCMeta

    def __str__(self):
        return 'instance of {}(Monitor)'.format(self.__class__)

    def __repr__(self):
        if hasattr(self, '_making_repr') and self._making_repr:
            return '{}(recursive reference)'.format(self.__class__)
        else:
            self._making_repr = True
            return_value = '{}({})'.format(
                self.__class__,
                '\n'.join('{}: {}'.format(repr(key), repr(value))
                          for key, value in self.__dict__.items()
                          if key != '_making_repr'))
            self._making_repr = False
            return return_value

    @abc.abstractmethod
    def store(self, state):
        """
        Stores the given state in the Monitor and performs class-specific
        actions.

        Args
        ----
        state: dict
            A model state dictionary.

        Raises
        ------
        InvalidStateError
            If state is not a valid input for the Diagnostic instance.
        """
