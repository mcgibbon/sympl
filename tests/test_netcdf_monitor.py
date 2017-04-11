import pytest
from sympl import NetCDFMonitor, DataArray, InvalidStateError
import os
from datetime import datetime, timedelta
import numpy as np
import xarray as xr

random = np.random.RandomState(0)

nx = 5
ny = 5
nz = 3
state = {
    'time': datetime(2013, 7, 20),
    'air_temperature': DataArray(
        random.randn(nx, ny, nz),
        dims=['lon', 'lat', 'mid_levels'],
        attrs={'units': 'degK'},
    ),
    'air_pressure': DataArray(
        random.randn(nx, ny, nz),
        dims=['lon', 'lat', 'mid_levels'],
        attrs={'units': 'Pa'},
    ),
}


def test_netcdf_monitor_initializes():
    assert not os.path.isfile('out.nc')
    NetCDFMonitor('out.nc')
    assert not os.path.isfile('out.nc')  # should not create output file on init


def test_netcdf_monitor_initializes_with_kwargs():
    assert not os.path.isfile('out.nc')
    NetCDFMonitor(
        'out.nc',
        time_units='hours',
        store_names=('air_temperature', 'air_pressure'),
        write_on_store=True
    )
    assert not os.path.isfile('out.nc')  # should not create output file on init


def test_netcdf_monitor_single_time_all_vars():
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc')
        monitor.store(state)
        assert not os.path.isfile('out.nc')  # not set to write on store
        monitor.write()
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 2
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (1, nx, ny, nz)
            assert 'air_pressure' in ds.data_vars.keys()
            assert ds.data_vars['air_pressure'].attrs['units'] == 'Pa'
            assert tuple(ds.data_vars['air_pressure'].shape) == (1, nx, ny, nz)
            assert len(ds['time']) == 1
            assert ds['time'][0] == np.datetime64(state['time'])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_multiple_times_batched_all_vars():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc')
        for time in time_list:
            current_state['time'] = time
            monitor.store(current_state)
            assert not os.path.isfile('out.nc')  # not set to write on store
        monitor.write()
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 2
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (
                len(time_list), nx, ny, nz)
            assert 'air_pressure' in ds.data_vars.keys()
            assert ds.data_vars['air_pressure'].attrs['units'] == 'Pa'
            assert tuple(ds.data_vars['air_pressure'].shape) == (
                len(time_list), nx, ny, nz)
            assert len(ds['time']) == len(time_list)
            assert np.all(
                ds['time'].values == [np.datetime64(time) for time in time_list])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_multiple_times_sequential_all_vars():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc')
        for time in time_list:
            current_state['time'] = time
            monitor.store(current_state)
            monitor.write()
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 2
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (
                len(time_list), nx, ny, nz)
            assert 'air_pressure' in ds.data_vars.keys()
            assert ds.data_vars['air_pressure'].attrs['units'] == 'Pa'
            assert tuple(ds.data_vars['air_pressure'].shape) == (
                len(time_list), nx, ny, nz)
            assert len(ds['time']) == len(time_list)
            assert np.all(
                ds['time'].values == [np.datetime64(time) for time in time_list])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_multiple_times_sequential_all_vars_timedelta():
    time_list = [
        timedelta(hours=0),
        timedelta(hours=6),
        timedelta(hours=12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc')
        for time in time_list:
            current_state['time'] = time
            monitor.store(current_state)
            monitor.write()
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 2
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (
                len(time_list), nx, ny, nz)
            assert 'air_pressure' in ds.data_vars.keys()
            assert ds.data_vars['air_pressure'].attrs['units'] == 'Pa'
            assert tuple(ds.data_vars['air_pressure'].shape) == (
                len(time_list), nx, ny, nz)
            assert len(ds['time']) == len(time_list)
            assert np.all(
                ds['time'].values == [np.timedelta64(time) for time in time_list])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')

def test_netcdf_monitor_multiple_times_batched_single_var():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc', store_names=['air_temperature'])
        for time in time_list:
            current_state['time'] = time
            monitor.store(current_state)
            assert not os.path.isfile('out.nc')  # not set to write on store
        monitor.write()
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 1
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (
                len(time_list), nx, ny, nz)
            assert len(ds['time']) == len(time_list)
            assert np.all(
                ds['time'].values == [np.datetime64(time) for time in time_list])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_multiple_times_sequential_single_var():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc', store_names=['air_temperature'])
        for time in time_list:
            current_state['time'] = time
            monitor.store(current_state)
            monitor.write()
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 1
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (
                len(time_list), nx, ny, nz)
            assert len(ds['time']) == len(time_list)
            assert np.all(
                ds['time'].values == [np.datetime64(time) for time in time_list])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_single_write_on_store():
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc', write_on_store=True)
        monitor.store(state)
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 2
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (1, nx, ny, nz)
            assert 'air_pressure' in ds.data_vars.keys()
            assert ds.data_vars['air_pressure'].attrs['units'] == 'Pa'
            assert tuple(ds.data_vars['air_pressure'].shape) == (1, nx, ny, nz)
            assert len(ds['time']) == 1
            assert ds['time'][0] == np.datetime64(state['time'])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_multiple_write_on_store():
    time_list = [
        datetime(2013, 7, 20, 0),
        datetime(2013, 7, 20, 6),
        datetime(2013, 7, 20, 12),
    ]
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc', write_on_store=True)
        for time in time_list:
            current_state['time'] = time
            monitor.store(current_state)
        assert os.path.isfile('out.nc')
        with xr.open_dataset('out.nc') as ds:
            assert len(ds.data_vars.keys()) == 2
            assert 'air_temperature' in ds.data_vars.keys()
            assert ds.data_vars['air_temperature'].attrs['units'] == 'degK'
            assert tuple(ds.data_vars['air_temperature'].shape) == (
                len(time_list), nx, ny, nz)
            assert 'air_pressure' in ds.data_vars.keys()
            assert ds.data_vars['air_pressure'].attrs['units'] == 'Pa'
            assert tuple(ds.data_vars['air_pressure'].shape) == (
                len(time_list), nx, ny, nz)
            assert len(ds['time']) == len(time_list)
            assert np.all(
                ds['time'].values == [np.datetime64(time) for time in time_list])
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_raises_when_names_change_on_sequential_write():
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc')
        current_state['time'] = datetime(2013, 7, 20, 0)
        monitor.store(current_state)
        monitor.write()
        assert os.path.isfile('out.nc')
        current_state['time'] = datetime(2013, 7, 20, 6)
        current_state['air_density'] = current_state['air_pressure']
        monitor.store(current_state)
        try:
            monitor.write()
        except InvalidStateError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError(
                'Expected InvalidStateError but was not raised.')
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')


def test_netcdf_monitor_raises_when_names_change_on_batch_write():
    current_state = state.copy()
    try:
        assert not os.path.isfile('out.nc')
        monitor = NetCDFMonitor('out.nc')
        current_state['time'] = datetime(2013, 7, 20, 0)
        monitor.store(current_state)
        assert not os.path.isfile('out.nc')
        current_state['time'] = datetime(2013, 7, 20, 6)
        current_state['air_density'] = current_state['air_pressure']
        monitor.store(current_state)
        try:
            monitor.write()
        except InvalidStateError:
            pass
        except Exception as err:
            raise err
        else:
            raise AssertionError(
                'Expected InvalidStateError but was not raised.')
    finally:  # make sure we remove the output file
        if os.path.isfile('out.nc'):
            os.remove('out.nc')

if __name__ == '__main__':
    pytest.main([__file__])
