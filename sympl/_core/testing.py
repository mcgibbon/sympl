import abc
import os
from glob import glob
import xarray as xr
from .util import same_list
from .base_components import Diagnostic, Prognostic, Implicit
from .timestepping import TimeStepper
import numpy as np
from .units import is_valid_unit
from datetime import timedelta


def cache_dictionary(dictionary, filename):
    dataset = xr.Dataset(dictionary)
    dataset.to_netcdf(filename, engine='scipy')


def load_dictionary(filename):
    dataset = xr.open_dataset(filename, engine='scipy')
    return dict(dataset.data_vars)


def call_with_timestep_if_needed(
        component, state, timestep=timedelta(seconds=10.)):
    if isinstance(component, (Implicit, TimeStepper)):
        return component(state, timestep=timestep)
    else:
        return component(state)


def compare_outputs(current, cached):
    if isinstance(current, tuple) and isinstance(cached, tuple):
        for i in range(len(current)):
            compare_one_state_pair(current[i], cached[i])
    elif (not isinstance(current, tuple)) and (not isinstance(cached, tuple)):
        compare_one_state_pair(current, cached)
    else:
        raise AssertionError('Different number of dicts returned than cached.')


def compare_one_state_pair(current, cached):
    for key in current.keys():
        try:
            assert np.all(np.isclose(current[key].values, cached[key].values))
            for attr in current[key].attrs:
                assert current[key].attrs[attr] == cached[key].attrs[attr]
            for attr in cached[key].attrs:
                assert attr in current[key].attrs
            assert current[key].dims == cached[key].dims
        except AssertionError as err:
            raise AssertionError('Error for {}: {}'.format(key, err))
    for key in cached.keys():
        assert key in current.keys()


def assert_dimension_lengths_are_consistent(state):
    dimension_lengths = {}
    for name, value in state.items():
        if name != 'time':
            for i, dim_name in enumerate(value.dims):
                try:
                    if dim_name in dimension_lengths:
                        assert dimension_lengths[dim_name] == value.shape[i]
                    else:
                        dimension_lengths[dim_name] = value.shape[i]
                except AssertionError as err:
                    raise AssertionError(
                        'Inconsistent length on dimension {} for value {}:'
                        '{}'.format(dim_name, name, err))


class ComponentTestBase(object):

    cache_folder = None

    @abc.abstractmethod
    def get_input_state(self):
        pass

    @abc.abstractmethod
    def get_component_instance(self):
        pass

    def get_cached_output(self,):
        cache_filename_list = sorted(glob(
            os.path.join(
                self.cache_folder,
                '{}-*.cache'.format(
                    self.__class__.__name__))))
        if len(cache_filename_list) > 0:
            return_list = []
            for filename in cache_filename_list:
                return_list.append(load_dictionary(filename))
            if len(return_list) > 1:
                return tuple(return_list)
            elif len(return_list) == 1:
                return return_list[0]
        else:
            return None

    def cache_output(self, output):
        if not isinstance(output, tuple):
            output = (output,)
        for i in range(len(output)):
            cache_filename = os.path.join(
                self.cache_folder, '{}-{}.cache'.format(self.__class__.__name__, i))
            cache_dictionary(output[i], cache_filename)

    def test_output_matches_cached_output(self):
        state = self.get_input_state()
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, state)
        cached_output = self.get_cached_output()
        if cached_output is None:
            self.cache_output(output)
            raise AssertionError(
                'Failed due to no cached output, cached current output')
        else:
            compare_outputs(output, cached_output)

    def test_component_listed_inputs_are_accurate(self):
        state = self.get_input_state()
        component = self.get_component_instance()
        input_state = {}
        for key in component.inputs:
            input_state[key] = state[key]
        output = call_with_timestep_if_needed(component, state)
        cached_output = self.get_cached_output()
        if cached_output is not None:
            compare_outputs(output, cached_output)

    def test_inputs_and_outputs_have_consistent_dim_lengths(self):
        """A given dimension name should always have the same length."""
        input_state = self.get_input_state()
        assert_dimension_lengths_are_consistent(input_state)
        component = self.get_component_instance()
        output = call_with_timestep_if_needed(component, input_state)
        if isinstance(output, tuple):
            # Check diagnostics/tendencies/outputs are consistent with one
            # another
            test_state = {}
            for state in output:
                test_state.update(state)
            assert_dimension_lengths_are_consistent(test_state)
        else:
            test_state = output  # if not a tuple assume it's a dict
            assert_dimension_lengths_are_consistent(test_state)

    def test_listed_outputs_are_accurate(self):
        state = self.get_input_state()
        component = self.get_component_instance()
        if isinstance(component, Diagnostic):
            diagnostics = component(state)
            assert same_list(component.diagnostics, diagnostics.keys())
        elif isinstance(component, Prognostic):
            tendencies, diagnostics = component(state)
            assert same_list(component.tendencies, tendencies.keys())
            assert same_list(component.diagnostics, diagnostics.keys())
        elif isinstance(component, Implicit):
            diagnostics, new_state = component(state)
            assert same_list(component.diagnostics, diagnostics.keys())
            assert same_list(component.outputs, new_state.keys())

    def test_modifies_attribute_is_accurate(self):
        state = self.get_input_state()
        component = self.get_component_instance()
        if not hasattr(component, 'modifies'):
            raise AssertionError("component does not have a 'modifies' property")
        original_state = {}
        for key, value in state.items():
            if key == 'time':
                original_state[key] = state[key]
            else:
                original_state[key] = state[key].copy(deep=True)
        component(state)
        for key in state.keys():
            if key not in state.modifies:
                assert np.all(original_state[key] == state[key]), key

    def test_has_input_properties(self):
        component = self.get_component_instance()
        assert hasattr(component, 'input_properties')
        assert isinstance(component.input_properties, dict)

    def test_has_output_properties(self):
        component = self.get_component_instance()
        if isinstance(component, [Implicit, TimeStepper]):
            assert hasattr(component, 'output_properties')
            assert isinstance(component.output_properties, dict)

    def test_has_tendency_properties(self):
        component = self.get_component_instance()
        if isinstance(component, Prognostic):
            assert hasattr(component, 'tendency_properties')
            assert isinstance(component.tendency_properties, dict)

    def test_has_diagnostic_properties(self):
        component = self.get_component_instance()
        if isinstance(component, [Diagnostic, Prognostic, Implicit, TimeStepper]):
            assert hasattr(component, 'diagnostic_properties')
            assert isinstance(component.diagnostic_properties, dict)

    def test_input_unit_properties_are_valid(self):
        component = self.get_component_instance()
        for name, properties in component.input_properties.items():
            if 'units' not in properties.keys():
                raise AssertionError(
                    "quantity {} has no 'units' property defined".format(name))
            if not is_valid_unit(properties['units']):
                raise AssertionError(
                    "unit {} for quantity {} is not recognized".format(
                        properties['units'], name
                    )
                )

    def test_diagnostic_unit_properties_are_valid(self):
        component = self.get_component_instance()
        if hasattr(component, 'diagnostic_properties'):
            for name, properties in component.diagnostic_properties.items():
                if 'units' not in properties.keys():
                    raise AssertionError(
                        "quantity {} has no 'units' property defined in "
                        "diagnostic_properties".format(name))
                if not is_valid_unit(properties['units']):
                    raise AssertionError(
                        "unit {} for quantity {} in diagnostic_properties "
                        "is not recognized".format(
                            properties['units'], name
                        )
                    )

    def test_tendency_unit_properties_are_valid(self):
        component = self.get_component_instance()
        if hasattr(component, 'tendency_properties'):
            for name, properties in component.tendency_properties.items():
                if 'units' not in properties.keys():
                    raise AssertionError(
                        "quantity {} has no 'units' property defined in "
                        "tendency_properties".format(name))
                if not is_valid_unit(properties['units']):
                    raise AssertionError(
                        "unit {} for quantity {} in tendency_properties "
                        "is not recognized".format(
                            properties['units'], name
                        )
                    )

    def test_output_unit_properties_are_valid(self):
        component = self.get_component_instance()
        if hasattr(component, 'output_properties'):
            for name, properties in component.output_properties.items():
                if 'units' not in properties.keys():
                    raise AssertionError(
                        "quantity {} has no 'units' property defined in "
                        "output_properties".format(name))
                if not is_valid_unit(properties['units']):
                    raise AssertionError(
                        "unit {} for quantity {} in output_properties "
                        "is not recognized".format(
                            properties['units'], name
                        )
                    )

    def test_tendency_dims_like_properties_are_inputs(self):
        component = self.get_component_instance()
        if hasattr(component, 'tendency_properties'):
            for name, properties in component.tendency_properties.items():
                if 'dims_like' not in properties.keys():
                    raise AssertionError(
                        "quantity {} has no 'dims_like' property defined in "
                        "tendency_properties".format(name))
                if properties['dims_like'] not in component.inputs:
                    raise AssertionError(
                        'quantity {} has dims_like {} in tendency_properties, '
                        'but {} is not specified as an input'.format(
                            name, properties['dims_like'],
                            properties['dims_like'])
                    )

    def test_diagnostic_dims_like_properties_are_inputs(self):
        component = self.get_component_instance()
        if hasattr(component, 'diagnostic_properties'):
            for name, properties in component.diagnostic_properties.items():
                if 'dims_like' not in properties.keys():
                    raise AssertionError(
                        "quantity {} has no 'dims_like' property defined in "
                        "diagnostic_properties".format(name))
                if properties['dims_like'] not in component.inputs:
                    raise AssertionError(
                        'quantity {} has dims_like {} in diagnostic_properties, '
                        'but {} is not specified as an input'.format(
                            name, properties['dims_like'],
                            properties['dims_like'])
                    )

    def test_output_dims_like_properties_are_inputs(self):
        component = self.get_component_instance()
        if hasattr(component, 'output_properties'):
            for name, properties in component.output_properties.items():
                if 'dims_like' not in properties.keys():
                    raise AssertionError(
                        "quantity {} has no 'dims_like' property defined in "
                        "output_properties".format(name))
                if properties['dims_like'] not in component.inputs:
                    raise AssertionError(
                        'quantity {} has dims_like {} in output_properties, '
                        'but {} is not specified as an input'.format(
                            name, properties['dims_like'],
                            properties['dims_like'])
                    )
