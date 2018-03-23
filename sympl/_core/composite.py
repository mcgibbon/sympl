from .exceptions import SharedKeyError
from .base_components import Prognostic, Diagnostic, Monitor
from .util import (
    update_dict_by_adding_another, ensure_no_shared_keys,
    combine_component_properties)


class ComponentComposite(object):
    """
    A composite of components that allows them to be called as one object.

    Attributes
    ----------
    component_list: list
        The components being composited by this object.
    """

    component_class = None

    @property
    def input_properties(self):
        return combine_component_properties(self.component_list, 'input_properties')

    @property
    def diagnostic_properties(self):
        return_dict = {}
        for component in self.component_list:
            return_dict.update(component.diagnostic_properties)
        return return_dict

    def __str__(self):
        return '{}(\n{}\n)'.format(
            self.__class__,
            ',\n'.join(str(component) for component in self.component_list))

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__,
            ',\n'.join(repr(component) for component in self.component_list))

    def __init__(self, *args):
        """
        Args
        ----
        *args
            The components that should be wrapped by this object.

        Raises
        ------
        SharedKeyError
            If two components compute the same diagnostic quantity.
        """
        if self.component_class is not None:
            ensure_components_have_class(args, self.component_class)
        self.component_list = args

    def ensure_no_diagnostic_output_overlap(self):
        diagnostic_names = []
        for component in self.component_list:
            diagnostic_names.extend(component.diagnostic_properties.keys())
        for name in diagnostic_names:
            if diagnostic_names.count(name) > 1:
                raise SharedKeyError(
                    'Two components in a composite should not compute '
                    'the same diagnostic, but multiple passed '
                    'components compute {}'.format(name))

    def _combine_attribute(self, attr):
        return_attr = []
        for component in self.component_list:
            return_attr.extend(getattr(component, attr))
        return tuple(set(return_attr))  # set to deduplicate


def ensure_components_have_class(components, component_class):
    for component in components:
        if not isinstance(component, component_class):
            raise TypeError(
                'Component should be of type {} but is type {}'.format(
                    component_class, type(component)))


class PrognosticComposite(ComponentComposite, Prognostic):

    component_class = Prognostic

    @property
    def tendency_properties(self):
        return combine_component_properties(self.component_list, 'tendency_properties')

    def __init__(self, *args):
        super(PrognosticComposite, self).__init__(*args)
        self.ensure_tendency_outputs_are_compatible()
        self.ensure_no_diagnostic_output_overlap()

    def ensure_tendency_outputs_are_compatible(self):
        self.tendency_properties

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
        SharedKeyError
            If multiple Prognostic objects contained in the
            collection return the same diagnostic quantity.
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for a Prognostic instance.
        """
        return_tendencies = {}
        return_diagnostics = {}
        for prognostic in self.component_list:
            tendencies, diagnostics = prognostic(state)
            update_dict_by_adding_another(return_tendencies, tendencies)
            return_diagnostics.update(diagnostics)
        return return_tendencies, return_diagnostics

    def array_call(self, state):
        raise NotImplementedError()


class DiagnosticComposite(ComponentComposite, Diagnostic):

    component_class = Diagnostic

    def __init__(self, *args):
        super(DiagnosticComposite, self).__init__(*args)
        self.ensure_no_diagnostic_output_overlap()

    def __call__(self, state):
        """
        Gets diagnostics from the passed model state.

        Args
        ----
        state : dict
            A model state dictionary.

        Returns
        -------
        diagnostics: dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.

        Raises
        ------
        SharedKeyError
            If multiple Diagnostic objects contained in the
            collection return the same diagnostic quantity.
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for a Diagnostic instance.
        """
        return_diagnostics = {}
        for diagnostic_component in self.component_list:
            diagnostics = diagnostic_component(state)
            # ensure two diagnostics don't compute the same quantity
            ensure_no_shared_keys(return_diagnostics, diagnostics)
            return_diagnostics.update(diagnostics)
        return return_diagnostics

    def array_call(self, state):
        raise NotImplementedError()


class MonitorComposite(ComponentComposite):

    component_class = Monitor

    def store(self, state):
        """
        Stores the given state in the Monitor and performs class-specific
        actions.

        Args
        ----
        state : dict
            A model state dictionary.

        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for a Monitor instance.
        """
        for monitor in self.component_list:
            monitor.store(state)
