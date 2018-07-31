def copy_untouched_quantities(old_state, new_state):
    for key in old_state.keys():
        if key not in new_state:
            new_state[key] = old_state[key]


def add(state_1, state_2):
    out_state = {}
    if 'time' in state_1.keys():
        out_state['time'] = state_1['time']
    for key in state_1.keys():
        if key != 'time':
            out_state[key] = state_1[key] + state_2[key]
            if hasattr(out_state[key], 'attrs'):
                out_state[key].attrs = state_1[key].attrs
    return out_state


def multiply(scalar, state):
    out_state = {}
    if 'time' in state.keys():
        out_state['time'] = state['time']
    for key in state.keys():
        if key != 'time':
            out_state[key] = scalar * state[key]
            if hasattr(out_state[key], 'attrs'):
                out_state[key].attrs = state[key].attrs
    return out_state
