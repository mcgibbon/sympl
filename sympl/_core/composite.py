from .base_components import TendencyComponent, DiagnosticComponent, Monitor, ImplicitTendencyComponent
from .util import (
    update_dict_by_adding_another, ensure_no_shared_keys)
from .combine_properties import combine_component_properties
from .exceptions import InvalidPropertyDictError


class InputPropertiesCompositeMixin(object):

    @property
    def input_properties(self):
        return combine_component_properties(self.component_list, 'input_properties')

    def __init__(self, *args):
        self.input_properties
        super(InputPropertiesCompositeMixin, self).__init__()

    def _combine_attribute(self, attr):
        return_attr = []
        for component in self.component_list:
            return_attr.extend(getattr(component, attr))
        return tuple(set(return_attr))  # set to deduplicate


class DiagnosticPropertiesCompositeMixin(object):

    @property
    def diagnostic_properties(self):
        return_dict = {}
        for component in self.component_list:
            ensure_no_shared_keys(component.diagnostic_properties, return_dict)
            return_dict.update(component.diagnostic_properties)
        for name, properties in return_dict.items():
            if 'dims' not in properties.keys() and not (name in self.input_properties):
                raise InvalidPropertyDictError(
                    'Must define dims for diagnostic output {}'.format(name))
        return return_dict

    def __init__(self, *args):
        self.diagnostic_properties
        super(DiagnosticPropertiesCompositeMixin, self).__init__()

    def _combine_attribute(self, attr):
        return_attr = []
        for component in self.component_list:
            return_attr.extend(getattr(component, attr))
        return tuple(set(return_attr))  # set to deduplicate


class ComponentComposite(object):
    """
    A composite of components that allows them to be called as one object.

    Attributes
    ----------
    component_list: list
        The components being composited by this object.
    """

    component_class = None

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
        InvalidPropertyDictError
            If two components require the same input or compute the same
            output quantity, and their dimensions or units are incompatible
            with one another.
        """
        if self.component_class is not None:
            ensure_components_have_class(args, self.component_class)
        self.component_list = args
        super(ComponentComposite, self).__init__()


def ensure_components_have_class(components, component_class):
    for component in components:
        if not isinstance(component, component_class):
            raise TypeError(
                'Component should be of type {} but is type {}'.format(
                    component_class, type(component)))


class TendencyComponentComposite(
        ComponentComposite, InputPropertiesCompositeMixin,
        DiagnosticPropertiesCompositeMixin, TendencyComponent):

    component_class = TendencyComponent

    @property
    def tendency_properties(self):
        return combine_component_properties(
            self.component_list, 'tendency_properties', self.input_properties)

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
        InvalidPropertyDictError
            If two components require the same input or compute the same
            output quantity, and their dimensions or units are incompatible
            with one another.
        """
        super(TendencyComponentComposite, self).__init__(*args)
        self.input_properties
        self.tendency_properties
        self.diagnostic_properties

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
            If state is not a valid input for a TendencyComponent instance.
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


class ImplicitTendencyComponentComposite(
        ComponentComposite, InputPropertiesCompositeMixin,
        DiagnosticPropertiesCompositeMixin, ImplicitTendencyComponent):

    component_class = (TendencyComponent, ImplicitTendencyComponent)

    @property
    def tendency_properties(self):
        return combine_component_properties(
            self.component_list, 'tendency_properties', self.input_properties)

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
        InvalidPropertyDictError
            If two components require the same input or compute the same
            output quantity, and their dimensions or units are incompatible
            with one another.
        """
        super(ImplicitTendencyComponentComposite, self).__init__(*args)
        self.input_properties
        self.tendency_properties
        self.diagnostic_properties

    def __call__(self, state, timestep):
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
            If state is not a valid input for a TendencyComponent instance.
        """
        return_tendencies = {}
        return_diagnostics = {}
        for prognostic in self.component_list:
            if isinstance(prognostic, ImplicitTendencyComponent):
                tendencies, diagnostics = prognostic(state, timestep)
            elif isinstance(prognostic, TendencyComponent):
                tendencies, diagnostics = prognostic(state)
            update_dict_by_adding_another(return_tendencies, tendencies)
            return_diagnostics.update(diagnostics)
        return return_tendencies, return_diagnostics

    def array_call(self, state):
        raise NotImplementedError()


class DiagnosticComponentComposite(
        ComponentComposite, InputPropertiesCompositeMixin,
        DiagnosticPropertiesCompositeMixin, DiagnosticComponent):

    component_class = DiagnosticComponent

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
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for a DiagnosticComponent instance.
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
