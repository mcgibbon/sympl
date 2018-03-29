import abc
from .composite import PrognosticComposite
from .time import timedelta
from .util import combine_component_properties, combine_properties
from .units import clean_units
from .state import copy_untouched_quantities
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
    name : string
        A label to be used for this object, for example as would be used for
        Y in the name "X_tendency_from_Y".
    """
    __metaclass__ = abc.ABCMeta

    time_unit_name = 's'
    time_unit_timedelta = timedelta(seconds=1)

    @property
    def input_properties(self):
        input_properties = combine_component_properties(
            self.prognostic_list, 'input_properties')
        return combine_properties([input_properties, self.output_properties])

    @property
    def diagnostic_properties(self):
        return_value = {}
        for prognostic in self.prognostic_list:
            return_value.update(prognostic.diagnostic_properties)
        if self.tendencies_in_diagnostics:
            tendency_properties = combine_component_properties(
                self.prognostic_list, 'tendency_properties')
            self._insert_tendencies_to_diagnostic_properties(
                return_value, tendency_properties)
        return return_value

    def _insert_tendencies_to_diagnostic_properties(
            self, diagnostic_properties, tendency_properties):
        for quantity_name, properties in tendency_properties.items():
            tendency_name = self._get_tendency_name(quantity_name)
            diagnostic_properties[tendency_name] = {
                'units': properties['units'],
                'dims': properties['dims'],
            }

    @property
    def output_properties(self):
        output_properties = combine_component_properties(
            self.prognostic_list, 'tendency_properties')
        for name, properties in output_properties.items():
            properties['units'] += ' {}'.format(self.time_unit_name)
            properties['units'] = clean_units(properties['units'])
        return output_properties

    @property
    def _tendency_properties(self):
        return combine_component_properties(self.prognostic_list, 'tendency_properties')

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

    def __init__(self, *args, **kwargs):
        """
        Initialize the TimeStepper.

        Parameters
        ----------
        *args : Prognostic
            Objects to call for tendencies when doing time stepping.
        tendencies_in_diagnostics : bool, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output. Default is False. If set to
            True, you probably want to give a name also.
        name : str
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name is used.
        """
        self.name = kwargs.pop('name', self.__class__.__name__)
        tendencies_in_diagnostics = kwargs.pop('tendencies_in_diagnostics', False)
        if len(kwargs) > 0:
            raise TypeError(
                "TimeStepper.__init__ got an unexpected keyword argument '{}'".format(
                    kwargs.popitem()[0]))
        if len(args) == 1 and isinstance(args[0], list):
            warnings.warn(
                'TimeSteppers should be given individual Prognostics rather '
                'than a list, and will not accept lists in a later version.')
            args = args[0]
        self._tendencies_in_diagnostics = tendencies_in_diagnostics
        self.prognostic = PrognosticComposite(*args)

    @property
    def prognostic_list(self):
        return self.prognostic.component_list

    @property
    def tendencies_in_diagnostics(self):  # value cannot be modified
        return self._tendencies_in_diagnostics

    def _get_tendency_name(self, quantity_name):
        return '{}_tendency_from_{}'.format(quantity_name, self.name)

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
        diagnostics, new_state = self._call(state, timestep)
        copy_untouched_quantities(state, new_state)
        if self.tendencies_in_diagnostics:
            self._insert_tendencies_to_diagnostics(
                state, new_state, timestep, diagnostics)
        return diagnostics, new_state

    def _insert_tendencies_to_diagnostics(
            self, state, new_state, timestep, diagnostics):
        output_properties = self.output_properties
        input_properties = self.input_properties
        for name in output_properties.keys():
            tendency_name = self._get_tendency_name(name)
            if tendency_name in diagnostics.keys():
                raise RuntimeError(
                    'A Prognostic has output tendencies as a diagnostic and has'
                    ' caused a name clash when trying to do so from this '
                    'TimeStepper ({}). You must disable '
                    'tendencies_in_diagnostics for this TimeStepper.'.format(
                        tendency_name))
            base_units = input_properties[name]['units']
            diagnostics[tendency_name] = (
                (new_state[name].to_units(base_units) - state[name].to_units(base_units)) /
                timestep.total_seconds() * self.time_unit_timedelta.total_seconds()
            )
            if base_units == '':
                diagnostics[tendency_name].attrs['units'] = '{}^-1'.format(
                    self.time_unit_name)
            else:
                diagnostics[tendency_name].attrs['units'] = '{} {}^-1'.format(
                    base_units, self.time_unit_name)

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
