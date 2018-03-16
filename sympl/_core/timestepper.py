import abc

from .composite import PrognosticComposite


class TimeStepper(object):
    """An object which integrates model state forward in time.

    It uses Prognostic and Diagnostic objects to update the current model state
    with diagnostics, and to return the model state at the next timestep.

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
    """

    __metaclass__ = abc.ABCMeta

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
            'instance of {}(TimeStepper)\n'
            '    inputs: {}\n'
            '    outputs: {}\n'
            '    diagnostics: {}\n'
            '    Prognostic components: {}'.format(
                self.__class__, self.input_properties.keys(),
                self.output_properties.keys(),
                self.diagnostic_properties.keys(),
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
