import abc
from .state import get_numpy_arrays_with_properties, restore_data_arrays_with_properties
from .time import timedelta
from .exceptions import (
    InvalidPropertyDictError, ComponentExtraOutputError,
    ComponentMissingOutputError, InvalidStateError)
from inspect import getargspec


def option_or_default(option, default):
    if option is None:
        return default
    else:
        return option


def apply_scale_factors(array_state, scale_factors):
    for key, factor in scale_factors.items():
        array_state[key] *= factor


def is_component_class(cls):
    return any(issubclass(cls, cls2) for cls2 in (Implicit, Prognostic, ImplicitPrognostic, Diagnostic))


def is_component_base_class(cls):
    return cls in (Implicit, Prognostic, ImplicitPrognostic, Diagnostic)


class ComponentMeta(abc.ABCMeta):

    def __instancecheck__(self, instance):
        if is_component_class(instance.__class__) or not is_component_base_class(self):
            return issubclass(instance.__class__, self)
        else:
            check_attributes = (
                'input_properties',
                'tendency_properties',
                'diagnostic_properties',
                'output_properties',
                '__call__',
                'array_call',
                'tendencies_in_diagnostics',
                'name',
            )
            required_attributes = list(
                att for att in check_attributes if hasattr(self, att)
            )
            disallowed_attributes = list(
                att for att in check_attributes if att not in required_attributes
            )
            if 'name' in disallowed_attributes:  # name is always allowed
                disallowed_attributes.remove('name')
            has_attributes = (
                all(hasattr(instance, att) for att in required_attributes) and
                not any(hasattr(instance, att) for att in disallowed_attributes)
            )
            if hasattr(self, '__call__') and not hasattr(instance, '__call__'):
                return False
            elif hasattr(self, '__call__'):
                timestep_in_class_call = 'timestep' in getargspec(self.__call__).args
                instance_argspec = getargspec(instance.__call__)
                timestep_in_instance_call = 'timestep' in instance_argspec.args
                instance_defaults = {}
                if instance_argspec.defaults is not None:
                    n = len(instance_argspec.args) - 1
                    for i, default in enumerate(reversed(instance_argspec.defaults)):
                        instance_defaults[instance_argspec.args[n-i]] = default
                timestep_optional = (
                    'timestep' in instance_defaults.keys() and instance_defaults['timestep'] is None)
                has_correct_spec = (timestep_in_class_call == timestep_in_instance_call) or timestep_optional
            else:
                raise RuntimeError(
                    'Cannot check instance type on component subclass that has '
                    'no __call__ method')
            return has_attributes and has_correct_spec


def check_overlapping_aliases(properties, properties_name):
    defined_aliases = set()
    for name, properties in properties.items():
        if 'alias' in properties.keys():
            if properties['alias'] not in defined_aliases:
                defined_aliases.add(properties['alias'])
            else:
                raise InvalidPropertyDictError(
                    'Multiple quantities map to alias {} in {} '
                    'properties'.format(
                        properties['alias'], properties_name)
                )


class InputMixin(object):

    def __init__(self):
        for name, properties in self.input_properties.items():
            if 'units' not in properties.keys():
                raise InvalidPropertyDictError(
                    'Input properties do not have units defined for {}'.format(name))
            if 'dims' not in properties.keys():
                raise InvalidPropertyDictError(
                    'Input properties do not have dims defined for {}'.format(name)
                )
        check_overlapping_aliases(self.input_properties, 'input')
        super(InputMixin, self).__init__()

    def _check_inputs(self, state):
        for key in self.input_properties.keys():
            if key not in state.keys():
                raise InvalidStateError('Missing input quantity {}'.format(key))


class TendencyMixin(object):

    def __init__(self):
        for name, properties in self.tendency_properties.items():
            if 'units' not in properties.keys():
                raise InvalidPropertyDictError(
                    'Tendency properties do not have units defined for {}'.format(name))
            if 'dims' not in properties.keys() and name not in self.input_properties.keys():
                raise InvalidPropertyDictError(
                    'Tendency properties do not have dims defined for {}'.format(name)
                )
        check_overlapping_aliases(self.tendency_properties, 'tendency')
        super(TendencyMixin, self).__init__()

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

    def __init__(self):
        for name, properties in self.diagnostic_properties.items():
            if 'units' not in properties.keys():
                raise InvalidPropertyDictError(
                    'Diagnostic properties do not have units defined for {}'.format(name))
            if 'dims' not in properties.keys() and name not in self.input_properties.keys():
                raise InvalidPropertyDictError(
                    'Diagnostic properties do not have dims defined for {}'.format(name)
                )
        check_overlapping_aliases(self.diagnostic_properties, 'diagnostic')
        super(DiagnosticMixin, self).__init__()

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
                'Component {} did not compute diagnostic(s) {}'.format(
                    self.__class__.__name__, ', '.join(missing_diagnostics)))

    def _check_extra_diagnostics(self, diagnostics_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_diagnostic_aliases.keys())
        for value_list in self._wanted_diagnostic_aliases.values():
            wanted_set.update(value_list)
        extra_diagnostics = set(diagnostics_dict.keys()).difference(wanted_set)
        if len(extra_diagnostics) > 0:
            raise ComponentExtraOutputError(
                'Component {} computed diagnostic(s) {} which are not in '
                'diagnostic_properties'.format(
                    self.__class__.__name__, ', '.join(extra_diagnostics)))

    def _check_diagnostics(self, diagnostics_dict):
        self._check_missing_diagnostics(diagnostics_dict)
        self._check_extra_diagnostics(diagnostics_dict)


class OutputMixin(object):

    def __init__(self):
        for name, properties in self.output_properties.items():
            if 'units' not in properties.keys():
                raise InvalidPropertyDictError(
                    'Output properties do not have units defined for {}'.format(name))
            if 'dims' not in properties.keys() and name not in self.input_properties.keys():
                raise InvalidPropertyDictError(
                    'Output properties do not have dims defined for {}'.format(name)
                )
        check_overlapping_aliases(self.output_properties, 'output')
        super(OutputMixin, self).__init__()

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
                'Component {} did not compute output(s) {}'.format(
                    self.__class__.__name__, ', '.join(missing_outputs)))

    def _check_extra_outputs(self, outputs_dict):
        wanted_set = set()
        wanted_set.update(self._wanted_output_aliases.keys())
        for value_list in self._wanted_output_aliases.values():
            wanted_set.update(value_list)
        extra_outputs = set(outputs_dict.keys()).difference(wanted_set)
        if len(extra_outputs) > 0:
            raise ComponentExtraOutputError(
                'Component {} computed output(s) {} which are not in '
                'output_properties'.format(
                    self.__class__.__name__, ', '.join(extra_outputs)))

    def _check_outputs(self, output_dict):
        self._check_missing_outputs(output_dict)
        self._check_extra_outputs(output_dict)


class Implicit(DiagnosticMixin, OutputMixin, InputMixin):
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
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    time_unit_name : str
        The unit to use for time differencing when putting tendencies in
        diagnostics.
    time_unit_timedelta: timedelta
        A timedelta corresponding to a single time unit as used for time
        differencing when putting tendencies in diagnostics.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """
    __metaclass__ = ComponentMeta

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

    def __init__(self, tendencies_in_diagnostics=False, name=None):
        """
        Initializes the Implicit object.

        Args
        ----
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output based on first order time
            differencing of output values.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__.lower()
        if tendencies_in_diagnostics:
            self._added_tendency_properties = self._insert_tendency_properties()
        else:
            self._added_tendency_properties = set()
        super(Implicit, self).__init__()

    def _insert_tendency_properties(self):
        added_names = []
        for name, properties in self.output_properties.items():
            tendency_name = self._get_tendency_name(name)
            if properties['units'] is '':
                units = 's^-1'
            else:
                units = '{} s^-1'.format(properties['units'])
            if 'dims' in properties.keys():
                dims = properties['dims']
            else:
                dims = self.input_properties[name]['dims']
            self.diagnostic_properties[tendency_name] = {
                'units': units,
                'dims': dims,
            }
            if name not in self.input_properties.keys():
                self.input_properties[name] = {
                    'dims': dims,
                    'units': properties['units'],
                }
            elif self.input_properties[name]['dims'] != dims:
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
        self._check_inputs(state)
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state['time'] = state['time']
        raw_diagnostics, raw_new_state = self.array_call(raw_state, timestep)
        self._check_diagnostics(raw_diagnostics)
        self._check_outputs(raw_new_state)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(
                raw_state, raw_new_state, timestep, raw_diagnostics)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties)
        new_state = restore_data_arrays_with_properties(
            raw_new_state, self.output_properties,
            state, self.input_properties)
        return diagnostics, new_state

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


class Prognostic(DiagnosticMixin, TendencyMixin, InputMixin):
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
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """
    __metaclass__ = ComponentMeta

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def tendency_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    name = None

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

    def __init__(self, tendencies_in_diagnostics=False, name=None):
        """
        Initializes the Implicit object.

        Args
        ----
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._added_diagnostic_names = []
        if self.tendencies_in_diagnostics:
            self._insert_tendency_properties()
        super(Prognostic, self).__init__()

    @property
    def tendencies_in_diagnostics(self):
        return self._tendencies_in_diagnostics

    def _insert_tendency_properties(self):
        for name, properties in self.tendency_properties.items():
            tendency_name = self._get_tendency_name(name)
            if 'dims' in properties.keys():
                dims = properties['dims']
            else:
                dims = self.input_properties[name]['dims']
            self.diagnostic_properties[tendency_name] = {
                'units': properties['units'],
                'dims': dims,
            }
            self._added_diagnostic_names.append(tendency_name)

    def _get_tendency_name(self, name):
        return '{}_tendency_from_{}'.format(name, self.name)

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
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state['time'] = state['time']
        raw_tendencies, raw_diagnostics = self.array_call(raw_state)
        self._check_tendencies(raw_tendencies)
        self._check_diagnostics(raw_diagnostics)
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties,
            ignore_names=self._added_diagnostic_names)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(tendencies, diagnostics)
        return tendencies, diagnostics

    def _insert_tendencies_to_diagnostics(self, tendencies, diagnostics):
        for name, value in tendencies.items():
            tendency_name = self._get_tendency_name(name)
            diagnostics[tendency_name] = value

    def _check_missing_diagnostics(self, diagnostics_dict):
        missing_diagnostics = set()
        for name, aliases in self._wanted_diagnostic_aliases.items():
            if (name not in diagnostics_dict.keys() and
                    name not in self._added_diagnostic_names and
                    not any(alias in diagnostics_dict.keys() for alias in aliases)):
                missing_diagnostics.add(name)
        if len(missing_diagnostics) > 0:
            raise ComponentMissingOutputError(
                'Component {} did not compute diagnostics {}'.format(
                    self.__class__.__name__, ', '.join(missing_diagnostics)))

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


class ImplicitPrognostic(DiagnosticMixin, TendencyMixin, InputMixin):
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
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output based on first order time
        differencing of output values.
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """
    __metaclass__ = ComponentMeta

    @abc.abstractproperty
    def input_properties(self):
        return {}

    @abc.abstractproperty
    def tendency_properties(self):
        return {}

    @abc.abstractproperty
    def diagnostic_properties(self):
        return {}

    name = None

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

    def __init__(self, tendencies_in_diagnostics=False, name=None):
        """
        Initializes the Implicit object.

        Args
        ----
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : string, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        """
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.name = name or self.__class__.__name__
        self._added_diagnostic_names = []
        if self.tendencies_in_diagnostics:
            self._insert_tendency_properties()
        super(ImplicitPrognostic, self).__init__()

    @property
    def tendencies_in_diagnostics(self):
        return self._tendencies_in_diagnostics

    def _insert_tendency_properties(self):
        for name, properties in self.tendency_properties.items():
            tendency_name = self._get_tendency_name(name)
            if 'dims' in properties.keys():
                dims = properties['dims']
            else:
                dims = self.input_properties[name]['dims']
            self.diagnostic_properties[tendency_name] = {
                'units': properties['units'],
                'dims': dims,
            }
            self._added_diagnostic_names.append(tendency_name)

    def _get_tendency_name(self, name):
        return '{}_tendency_from_{}'.format(name, self.name)

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
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state['time'] = state['time']
        raw_tendencies, raw_diagnostics = self.array_call(raw_state, timestep)
        self._check_tendencies(raw_tendencies)
        self._check_diagnostics(raw_diagnostics)
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties,
            ignore_names=self._added_diagnostic_names)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(tendencies, diagnostics)
        self._last_update_time = state['time']
        return tendencies, diagnostics

    def _insert_tendencies_to_diagnostics(self, tendencies, diagnostics):
        for name, value in tendencies.items():
            tendency_name = self._get_tendency_name(name)
            diagnostics[tendency_name] = value

    def _check_missing_diagnostics(self, diagnostics_dict):
        missing_diagnostics = set()
        for name, aliases in self._wanted_diagnostic_aliases.items():
            if (name not in diagnostics_dict.keys() and
                    name not in self._added_diagnostic_names and
                    not any(alias in diagnostics_dict.keys() for alias in aliases)):
                missing_diagnostics.add(name)
        if len(missing_diagnostics) > 0:
            raise ComponentMissingOutputError(
                'Component {} did not compute diagnostics {}'.format(
                    self.__class__.__name__, ', '.join(missing_diagnostics)))

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


class Diagnostic(DiagnosticMixin, InputMixin):
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
    """
    __metaclass__ = ComponentMeta

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

    def __init__(self):
        """
        Initializes the Implicit object.
        """
        super(Diagnostic, self).__init__()

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
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state['time'] = state['time']
        raw_diagnostics = self.array_call(raw_state)
        self._check_diagnostics(raw_diagnostics)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties)
        return diagnostics

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
