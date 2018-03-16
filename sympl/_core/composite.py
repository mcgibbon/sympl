from .exceptions import SharedKeyError
from .base_components import Prognostic, Diagnostic, Monitor
from sympl._core.util import update_dict_by_adding_another, ensure_no_shared_keys


class ComponentComposite(object):

    component_class = None

    def __str__(self):
        return '{}(\n{}\n)'.format(
            self.__class__,
            ',\n'.join(str(component) for component in self._components))

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__,
            ',\n'.join(repr(component) for component in self._components))

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
        self._components = args
        if hasattr(self, 'diagnostics'):
            if (len(self.diagnostics) !=
                    sum([len(comp.diagnostics) for comp in self._components])):
                raise SharedKeyError(
                    'Two components in a composite should not compute '
                    'the same diagnostic')

    def _combine_attribute(self, attr):
        return_attr = []
        for component in self._components:
            return_attr.extend(getattr(component, attr))
        return tuple(set(return_attr))  # set to deduplicate


def ensure_components_have_class(components, component_class):
    for component in components:
        for attr in ('input_properties', 'output_properties',
                     'diagnostic_properties', 'tendency_properties'):
            if hasattr(component_class, attr) and not hasattr(component, attr):
                raise TypeError(
                    'Component should have attribute {} but does not'.format(
                        attr))
            elif hasattr(component, attr) and not hasattr(component_class, attr):
                raise TypeError(
                    'Component should not have attribute {}, but does'.format(
                        attr))


class PrognosticComposite(ComponentComposite):
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
    """

    component_class = Prognostic

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
        for prognostic in self._components:
            tendencies, diagnostics = prognostic(state)
            update_dict_by_adding_another(return_tendencies, tendencies)
            return_diagnostics.update(diagnostics)
        return return_tendencies, return_diagnostics

    @property
    def inputs(self):
        return self._combine_attribute('inputs')

    @property
    def diagnostics(self):
        return self._combine_attribute('diagnostics')

    @property
    def tendencies(self):
        return self._combine_attribute('tendencies')


class DiagnosticComposite(ComponentComposite):
    """
    Attributes
    ----------
    inputs : tuple of str
        The quantities required in the state when the object is called.
    diagnostics : tuple of str
        The diagnostic quantities returned when the object is called.
    """

    component_class = Diagnostic

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
        for diagnostic_component in self._components:
            diagnostics = diagnostic_component(state)
            # ensure two diagnostics don't compute the same quantity
            ensure_no_shared_keys(return_diagnostics, diagnostics)
            return_diagnostics.update(diagnostics)
        return return_diagnostics

    @property
    def inputs(self):
        return self._combine_attribute('inputs')

    @property
    def diagnostics(self):
        return self._combine_attribute('diagnostics')


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
        for monitor in self._components:
            monitor.store(state)
