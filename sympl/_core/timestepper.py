import abc
from .composite import PrognosticComposite
from .time import timedelta
import warnings


class TimeStepper(object):
    """An object which integrates model state forward in time.

    It uses Prognostic and Diagnostic objects to update the current model state
    with diagnostics, and to return the model state at the next timestep.

    Attributes
    ----------
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
    prognostic : PrognosticComposite
        A composite of the Prognostic objects used by the TimeStepper
    prognostic_list: list of Prognostic
        A list of Prognostic objects called by the TimeStepper. These should
        be referenced when determining what inputs are necessary for the
        TimeStepper.
    tendencies_in_diagnostics : bool
        A boolean indicating whether this object will put tendencies of
        quantities in its diagnostic output.
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

    def diagnostic_properties(self):
        return_value = {}
        for prognostic in self.prognostic_list:
            return_value.update(prognostic.diagnostics)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostic_properties(return_value)
        return return_value

    @abc.abstractproperty
    def output_properties(self):
        return {}

    def __str__(self):
        return (
            'instance of {}(TimeStepper)\n'
            '    Prognostic components: {}'.format(self.prognostic_list)
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

    def __init__(self, *args, tendencies_in_diagnostics=False):
        """
        Initialize the TimeStepper.

        Parameters
        ----------
        *args : Prognostic
            Objects to call for tendencies when doing time stepping.
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        """
        if len(args) == 1 and isinstance(args[0], list):
            warnings.warn(
                'TimeSteppers should be given individual Prognostics rather '
                'than a list, and will not accept lists in a later version.')
            args = args[0]
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.prognostic = PrognosticComposite(*args)
        if tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostic_properties()

    def _insert_tendencies_to_diagnostic_properties(
            self, diagnostic_properties):
        for quantity_name, properties in self.output_properties.items():
            tendency_name = self._get_tendency_name(quantity_name, component_name)
            if properties['units'] is '':
                units = '{}^-1'.format(self.time_unit_name)
            else:
                units = '{} {}^-1'.format(
                    properties['units'], self.time_unit_name)
            diagnostic_properties[tendency_name] = {
                'units': units,
                'dims': properties['dims'],
            }

    def _insert_tendencies_to_diagnostics(
            self, raw_state, raw_new_state, timestep, raw_diagnostics):
        for name in self.output_properties.keys():
            tendency_name = self._get_tendency_name(name)
            raw_diagnostics[tendency_name] = (
                (raw_new_state[name] - raw_state[name]) /
                timestep.total_seconds() * self.time_unit_timedelta.total_seconds())

    @property
    def prognostic_list(self):
        return self.prognostic.component_list

    @property
    def tendencies_in_diagnostics(self):  # value cannot be modified
        return self._tendencies_in_diagnostics

    def _get_tendency_name(self, quantity_name, component_name):
        return '{}_tendency_from_{}'.format(quantity_name, component_name)

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

    @abc.abstractmethod
    def _call(self, state, timestep):
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
