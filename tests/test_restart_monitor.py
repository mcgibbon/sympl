import pytest
from sympl import RestartMonitor, DataArray, InvalidStateException
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


def test_restart_monitor_initializes():
    assert not os.path.isfile('restart.nc')
    RestartMonitor('restart.nc')
    assert not os.path.isfile('restart.nc')  # should not create file on init


def test_restart_monitor_stores_state():
    filename = 'restart.nc'
    assert not os.path.isfile(filename)
    monitor = RestartMonitor(filename)
    assert not os.path.isfile(filename)  # should not create file on init
    try:
        monitor.store(state)
        assert os.path.isfile(filename)
        new_monitor = RestartMonitor(filename)
        loaded_state = new_monitor.load()
    finally:
        if os.path.isfile(filename):
            os.remove(filename)
    for name in state.keys():
        if name is 'time':
            assert state['time'] == loaded_state['time']
        else:
            assert np.all(state[name].values == loaded_state[name].values)
            assert state[name].dims == loaded_state[name].dims
            assert state[name].attrs == loaded_state[name].attrs
    assert not os.path.isfile(filename)  # clean up should be successful

if __name__ == '__main__':
    pytest.main([__file__])
