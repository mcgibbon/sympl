import pytest
from sympl import RestartMonitor, DataArray, InvalidStateError
import os
from datetime import datetime, timedelta
import numpy as np

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
    restart_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'restart.nc')
    if os.path.isfile(restart_filename):
        os.remove(restart_filename)
    assert not os.path.isfile(restart_filename)
    RestartMonitor(restart_filename)
    assert not os.path.isfile(restart_filename)  # should not create file on init


def test_restart_monitor_stores_state():
    restart_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'restart.nc')
    if os.path.isfile(restart_filename):
        os.remove(restart_filename)
    assert not os.path.isfile(restart_filename)
    monitor = RestartMonitor(restart_filename)
    assert not os.path.isfile(restart_filename)  # should not create file on init
    monitor.store(state)
    assert os.path.isfile(restart_filename)
    new_monitor = RestartMonitor(restart_filename)
    loaded_state = new_monitor.load()
    for name in state.keys():
        if name is 'time':
            assert state['time'] == loaded_state['time']
        else:
            assert np.all(state[name].values == loaded_state[name].values)
            assert state[name].dims == loaded_state[name].dims
            assert state[name].attrs == loaded_state[name].attrs

if __name__ == '__main__':
    pytest.main([__file__])
