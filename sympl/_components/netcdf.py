from .._core.base_components import Monitor
from .._core.exceptions import (
    DependencyError, InvalidStateError)
from .._core.units import from_unit_to_another
from .._core.array import DataArray
from .._core.util import same_list, datetime64_to_datetime
import xarray as xr
import os
import numpy as np
from datetime import timedelta
try:
    import netCDF4 as nc4
except ImportError:
    nc4 = None

if nc4 is None:
    # If dependency is not installed, use a dummy object that will alert the
    # user they need to install the dependency if they try to use it
    class NetCDFMonitor(Monitor):

        def __init__(self, filename):
            raise DependencyError(
                'netCDF4-python must be installed to use NetCDFMonitor')

        def store(self, state):
            pass

else:
    class NetCDFMonitor(Monitor):
        """A Monitor which caches stored states and then writes them to a
        NetCDF file when requested."""

        def __init__(
                self, filename, time_units='seconds', store_names=None,
                write_on_store=False):
            """
            Args
            ----
            filename : str
                The file to which the NetCDF file will be written.
            time_units : str, optional
                The units in which time will be
                stored in the NetCDF file. Time is stored as an integer
                number of these units. Default is seconds.
            store_names : iterable of str, optional
                Names of quantities to store. If not given,
                all quantities are stored.
            write_on_store : bool, optional
                If True, stored changes are immediately written to file.
                This can result in many file open/close operations.
                Default is to write only when the write() method is
                called directly.
            """
            self._cached_state_dict = {}
            self._filename = filename
            self._time_units = time_units
            self._write_on_store = write_on_store
            if store_names is None:
                self._store_names = None
            else:
                self._store_names = ['time'] + list(store_names)

        def store(self, state):
            """
            Caches the given state. If write_on_store=True was passed on
            initialization, also writes to file. Normally a call to the
            write() method is required to write to file.

            Args
            ----
            state : dict
                A model state dictionary.

            Raises
            ------
            InvalidStateError
                If state is not a valid input for the Diagnostic instance.
            """
            if self._store_names is not None:
                name_list = set(state.keys()).intersection(self._store_names)
                cache_state = {name: state[name] for name in name_list}
            else:
                cache_state = state.copy()
            cache_state.pop('time')  # stored as key, not needed in state dict
            if state['time'] in self._cached_state_dict.keys():
                self._cached_state_dict[state['time']].update(cache_state)
            else:
                self._cached_state_dict[state['time']] = cache_state
            if self._write_on_store:
                self.write()

        @property
        def _write_mode(self):
            if not os.path.isfile(self._filename):
                return 'w'
            else:
                return 'a'

        def _ensure_cached_state_keys_compatible_with_dataset(self, dataset):
            file_keys = list(dataset.variables.keys())
            if 'time' in file_keys:
                file_keys.remove('time')
            if len(file_keys) > 0:
                self._ensure_cached_states_have_same_keys(file_keys)
            else:
                self._ensure_cached_states_have_same_keys()

        def _ensure_cached_states_have_same_keys(self, desired_keys=None):
            """
            Ensures all states in self._cached_state_dict have the same keys.
            If desired_keys is given, also ensure the keys are the same as
            the ones in desired_keys.

            Raises
            ------
            InvalidStateError
                If the cached states do not meet the requirements.
            """
            if len(self._cached_state_dict) == 0:
                return  # trivially true
            if desired_keys is not None:
                reference_keys = desired_keys
            else:
                reference_state = tuple(self._cached_state_dict.values())[0]
                reference_keys = reference_state.keys()
            for state in self._cached_state_dict.values():
                if not same_list(list(state.keys()), list(reference_keys)):
                    raise InvalidStateError(
                        'NetCDFMonitor was passed a different set of '
                        'quantities for different times: {} vs. {}'.format(
                            list(reference_keys), list(state.keys())))

        def _get_ordered_times_and_states(self):
            """Returns the items in self._cached_state_dict, sorted by time."""
            return zip(*sorted(self._cached_state_dict.items(), key=lambda x: x[0]))

        def write(self):
            """
            Write all cached states to the NetCDF file, and clear the cache.
            This will append to any existing NetCDF file.

            Raises
            ------
            InvalidStateError
                If cached states do not all have the same quantities
                as every other cached and written state.
            """
            with nc4.Dataset(self._filename, self._write_mode) as dataset:
                self._ensure_cached_state_keys_compatible_with_dataset(dataset)
                time_list, state_list = self._get_ordered_times_and_states()
                self._ensure_time_exists(dataset, time_list[0])
                it_start = dataset.dimensions['time'].size
                it_end = it_start + len(time_list)
                append_times_to_dataset(time_list, dataset, self._time_units)
                all_states = combine_states(state_list)
                for name, value in all_states.items():
                    ensure_variable_exists(dataset, name, value)
                    dataset.variables[name][
                        it_start:it_end, :] = value.values[:, :]
            self._cached_state_dict = {}

        def _ensure_time_exists(self, dataset, possible_reference_time):
            """Ensure an unlimited time dimension relevant to this monitor
            exists in the NetCDF4 dataset, and create it if it does not."""
            ensure_dimension_exists(dataset, 'time', None)
            if 'time' not in dataset.variables:
                dataset.createVariable('time', np.int64, ('time',))
                if isinstance(possible_reference_time, timedelta):
                    dataset.variables['time'].setncattr(
                        'units', self._time_units)
                else:  # assume datetime
                    dataset.variables['time'].setncattr(
                        'units', '{} since {}'.format(
                            self._time_units, possible_reference_time))
                    dataset.variables['time'].setncattr(
                        'calendar', 'proleptic_gregorian')


class RestartMonitor(Monitor):
    """
    A :py:class:`~sympl.Monitor` which stores model state in a NetCDF file,
    and can load that file back into the form of a model state.
    """

    def __init__(self, filename):
        self._filename = filename

    def store(self, state):
        """
        Write the state to the restart file, replacing any existing restart
        data.

        Parameters
        ----------
        state : dict
            A model state dictionary.
        """
        new_filename = self._filename + '.new'
        if os.path.isfile(new_filename):
            raise IOError('Filename {} already exists'.format(new_filename))
        netcdf_monitor = NetCDFMonitor(new_filename)
        netcdf_monitor.store(state)
        netcdf_monitor.write()

        if os.path.isfile(self._filename):
            os.rename(self._filename, self._filename + '.old')
        os.rename(new_filename, self._filename)
        if os.path.isfile(self._filename + '.old'):
            os.remove(self._filename + '.old')

    def load(self):
        """
        Load the state from the restart file.

        Returns
        -------
        state : dict
            The model state stored in the restart file.
        """
        dataset = xr.open_dataset(self._filename)
        state = {}
        for name, value in dataset.data_vars.items():
            state[name] = DataArray(value[0, :])  # remove time axis
        state['time'] = datetime64_to_datetime(dataset['time'][0])
        return state


def append_times_to_dataset(times, dataset, time_units):
    """Appends the given list of times to the dataset. Assumes the time units
    in the NetCDF4 dataset correspond to the string time_units."""
    it_start = dataset.dimensions['time'].size
    it_end = it_start + len(times)
    if isinstance(times[0], timedelta):
        times_list = []
        for time in times:
            times_list.append(time.total_seconds())
        time_array = from_unit_to_another(
            np.array(times_list), 'seconds', time_units)
        dataset.variables['time'][it_start:it_end] = time_array[:]
    else:  # assume datetime
        dataset.variables['time'][it_start:it_end] = nc4.date2num(
            times, dataset.variables['time'].units,
            calendar='proleptic_gregorian'
        )


def combine_states(states):
    """Takes in an iterable of state dictionaries, and combines them into a
    single returned state dictionary, adding a new first dimension to the
    DataArray values which corresponds to the order of the input state
    iterable."""
    return_dict = {}
    n_states = len(states)
    for name, value in states[0].items():
        return_dict[name] = DataArray(
            np.zeros((n_states,) + value.shape, dtype=value.values.dtype),
            dims=('time',) + value.dims, attrs=value.attrs)
    for i, state in enumerate(states):
        for name in state.keys():
            return_dict[name][i, :] = state[name][:]
    return return_dict


def ensure_variable_exists(dataset, name, data):
    """Dataset should be nc4.Dataset, name should be a string, and data should
    be a DataArray.

    Ensures there is a Variable in the dataset that corresponds to the given
    name and data, and creates it if not. Raises IOError if there is already
    a Variable but it is incompatible with the data."""
    if name not in dataset.variables:
        create_variable(dataset, name, data)
    else:
        ensure_variable_is_compatible(dataset.variables[name], name, data)


def create_variable(dataset, name, data):
    if isinstance(data, xr.DataArray):
        for i in range(len(data.dims)):
            try:
                if i == 0:  # time
                    ensure_dimension_exists(
                        dataset, data.dims[i], None)
                else:
                    ensure_dimension_exists(
                        dataset, data.dims[i], data.values.shape[i])
            except IOError as err:
                raise IOError(
                    'Error while creating {}: {}'.format(name, err))
        dataset.createVariable(
            name, data.values.dtype, data.dims)
        for key, value in data.attrs.items():
            dataset.variables[name].setncattr(key, value)
    else:
        raise TypeError('data must be of type DataArray')


def ensure_variable_is_compatible(variable, name, data):
    if variable.dimensions != data.dims:
        raise IOError(
            'Dimension in file is {} but on variable is {}'.format(
                variable.dimensions, data.dims))
    for key, value in data.attrs.items():
        if key not in variable.ncattrs():
            raise InvalidStateError(
                'State has attr {} for quantity {} but this is not '
                'present in the netCDF file'.format(key, name))
        elif value != variable.getncattr(key):
            raise InvalidStateError(
                'State has attr {} with value {} for quantity {} but '
                'the value in the netCDF file is {}'.format(
                    key, value, name,
                    variable.getncattr(key)))


def ensure_dimension_exists(dataset, dim_name, dim_length):
    if dim_name in dataset.dimensions:
        if dim_length is None:
            if not dataset.dimensions[dim_name].isunlimited():
                raise IOError(
                    'Dimension {} is unlimited in file but dim_length {} '
                    'is given'.format(dim_name, dim_length))
        elif dim_length != dataset.dimensions[dim_name].size:
            raise IOError(
                'Dimension {} is length {} in file but dim_length {} '
                'is given'.format(
                    dim_name, dataset.dimensions[dim_name].size, dim_length))
    else:
        dataset.createDimension(dim_name, dim_length)
