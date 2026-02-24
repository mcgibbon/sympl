import numpy as np
from sympl import Stepper, DataArray, datetime, timedelta
import time

class BenchmarkStepper(Stepper):
    input_properties = {}
    output_properties = {}
    diagnostic_properties = {}

    def __init__(self, properties):
        self.input_properties = properties
        self.output_properties = properties
        super().__init__()

    def array_call(self, state, timestep):
        new_state = {}
        for key in self.input_properties:
            new_state[key] = state[key] + 1.0
        return {}, new_state

def run_benchmark():
    nx, ny, nz = 50, 50, 50
    n_steps = 1000

    # Create state
    shape = (nx, ny, nz)
    dims = ('x', 'y', 'z')

    # Case 1: Matching dimensions (fast path)
    print(f"\nRunning benchmark (Matching dimensions) with shape {shape} for {n_steps} steps...")
    properties_matching = {
        'x_velocity': {'dims': ['x', 'y', 'z'], 'units': 'm s^-1'},
        'y_velocity': {'dims': ['x', 'y', 'z'], 'units': 'm s^-1'},
        'temperature': {'dims': ['x', 'y', 'z'], 'units': 'K'},
    }
    stepper_matching = BenchmarkStepper(properties_matching)

    state = {
        'time': datetime(2023, 1, 1),
        'x_velocity': DataArray(np.random.rand(*shape), dims=dims, attrs={'units': 'm s^-1'}),
        'y_velocity': DataArray(np.random.rand(*shape), dims=dims, attrs={'units': 'm s^-1'}),
        'temperature': DataArray(np.random.rand(*shape), dims=dims, attrs={'units': 'K'}),
    }
    timestep = timedelta(seconds=1)

    start_time = time.time()
    for _ in range(n_steps):
        _, new_state = stepper_matching(state, timestep)
        new_state['time'] = state['time'] + timestep
        state = new_state
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time: {duration:.4f} seconds")
    print(f"Time per step: {duration / n_steps:.6f} seconds")

    # Case 2: Permuted dimensions (transposition needed)
    print(f"\nRunning benchmark (Permuted dimensions) with shape {shape} for {n_steps} steps...")
    properties_permuted = {
        'x_velocity': {'dims': ['z', 'y', 'x'], 'units': 'm s^-1'},
        'y_velocity': {'dims': ['z', 'y', 'x'], 'units': 'm s^-1'},
        'temperature': {'dims': ['z', 'y', 'x'], 'units': 'K'},
    }
    stepper_permuted = BenchmarkStepper(properties_permuted)

    state = {
        'time': datetime(2023, 1, 1),
        'x_velocity': DataArray(np.random.rand(*shape), dims=dims, attrs={'units': 'm s^-1'}),
        'y_velocity': DataArray(np.random.rand(*shape), dims=dims, attrs={'units': 'm s^-1'}),
        'temperature': DataArray(np.random.rand(*shape), dims=dims, attrs={'units': 'K'}),
    }

    start_time = time.time()
    for _ in range(n_steps):
        _, new_state = stepper_permuted(state, timestep)
        new_state['time'] = state['time'] + timestep
        state = new_state
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time: {duration:.4f} seconds")
    print(f"Time per step: {duration / n_steps:.6f} seconds")

if __name__ == "__main__":
    run_benchmark()
