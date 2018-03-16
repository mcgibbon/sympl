import abc

from sympl._core.composite import PrognosticComposite


class Implicit(object):
    """
    Attributes
    ----------
    inputs : tuple of str
        The quantities required in the state when the object is called.
    diagnostics: tuple of str
        The quantities for which values for the old state are returned
        when the object is called.
    outputs: tuple of str
        The quantities for which values for the new state are returned
        when the object is called.
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
    """
    __metaclass__ = abc.ABCMeta

    input_properties = {}
    diagnostic_properties = {}
    output_properties = {}

    @property
    def inputs(self):
        return list(self.input_properties.keys())

    @property
    def diagnostics(self):
        return list(self.diagnostic_properties.keys())

    @property
    def outputs(self):
        return list(self.output_properties.keys())

    def __str__(self):
        return (
            'instance of {}(Implicit)\n'
            '    inputs: {}\n'
            '    outputs: {}\n'
            '    diagnostics: {}'.format(
                self.__class__, self.inputs, self.outputs, self.diagnostics)
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

    @abc.abstractmethod
    def __call__(self, state, timestep):
        """
        Gets diagnostics from the current model state and steps the state
        forward in time according to the timestep.

        Args
        ----
        state : dict
            A model state dictionary. Will be updated with any
            diagnostic quantities produced by this object for the time of
            the input state.
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


class Prognostic(object):
    """
    Attributes
    ----------
    inputs : tuple of str
        The quantities required in the state when the object is called.
    tendencies : tuple of str
        The quantities for which tendencies are returned when
        the object is called.
    diagnostics : tuple of str
        The diagnostic quantities returned when the object is called.
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
    """
    __metaclass__ = abc.ABCMeta

    input_properties = {}
    tendency_properties = {}
    diagnostic_properties = {}

    @property
    def inputs(self):
        return list(self.input_properties.keys())

    @property
    def tendencies(self):
        return list(self.tendency_properties.keys())

    @property
    def diagnostics(self):
        return list(self.diagnostic_properties.keys())

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

    @abc.abstractmethod
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


class ImplicitPrognostic(object):
    """
    Attributes
    ----------
    inputs : tuple of str
        The quantities required in the state when the object is called.
    tendencies : tuple of str
        The quantities for which tendencies are returned when
        the object is called.
    diagnostics : tuple of str
        The diagnostic quantities returned when the object is called.
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
    """
    __metaclass__ = abc.ABCMeta

    input_properties = {}
    tendency_properties = {}
    diagnostic_properties = {}

    @property
    def inputs(self):
        return list(self.input_properties.keys())

    @property
    def tendencies(self):
        return list(self.tendency_properties.keys())

    @property
    def diagnostics(self):
        return list(self.diagnostic_properties.keys())

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

    @abc.abstractmethod
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


class Diagnostic(object):
    """
    Attributes
    ----------
    inputs : tuple of str
        The quantities required in the state when the object is called.
    diagnostics : tuple of str
        The diagnostic quantities returned when the object is called.
    input_properties : dict
        A dictionary whose keys are quantities required in the state when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    diagnostic_properties : dict
        A dictionary whose keys are diagnostic quantities returned when the
        object is called, and values are dictionaries which indicate 'dims' and
        'units'.
    """
    __metaclass__ = abc.ABCMeta

    input_properties = {}
    diagnostic_properties = {}

    @property
    def inputs(self):
        return list(self.input_properties.keys())

    @property
    def diagnostics(self):
        return list(self.diagnostic_properties.keys())

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

    @abc.abstractmethod
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


class TimeStepper(object):
    """An object which integrates model state forward in time.

    It uses Prognostic and Diagnostic objects to update the current model state
    with diagnostics, and to return the model state at the next timestep.

    Attributes
    ----------
    inputs : tuple of str
        The quantities required in the state when the object is called.
    diagnostics: tuple of str
        The quantities for which values for the old state are returned
        when the object is called.
    outputs: tuple of str
        The quantities for which values for the new state are returned
        when the object is called.
    """

    __metaclass__ = abc.ABCMeta

    def __str__(self):
        return (
            'instance of {}(TimeStepper)\n'
            '    inputs: {}\n'
            '    outputs: {}\n'
            '    diagnostics: {}\n'
            '    Prognostic components: {}'.format(
                self.__class__, self.inputs, self.outputs, self.diagnostics,
                str(self._prognostic))
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

    def __init__(self, prognostic_list, **kwargs):
        self._prognostic = PrognosticComposite(*prognostic_list)

    @abc.abstractmethod
    def __call__(self, state, timestep):
        """
        Retrieves any diagnostics and returns a new state corresponding
        to the next timestep.

        Args
        ----
        state : dict
            The current model state.
        timestep : timedelta
            The amount of time to step forward.

        Returns
        -------
        diagnostics : dict
            Diagnostics from the timestep of the input state.
        new_state : dict
            The model state at the next timestep.
        """

    def _copy_untouched_quantities(self, old_state, new_state):
        for key in old_state.keys():
            if key not in new_state:
                new_state[key] = old_state[key]

    @property
    def inputs(self):
        return self._prognostic.inputs

    @property
    def outputs(self):
        return self._prognostic.tendencies

    @property
    def diagnostics(self):
        return self._prognostic.diagnostics
