from datetime import timedelta, datetime
from sympl import (
    Prognostic, UpdateFrequencyWrapper, TendencyInDiagnosticsWrapper)


class MockPrognostic(Prognostic):

    def __init__(self):
        self._num_updates = 0

    def __call__(self, state):
        self._num_updates += 1
        return {}, {'num_updates': self._num_updates}


def test_set_prognostic_update_frequency_calls_initially():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic(state)
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_caches_result():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic(state)
    tendencies, diagnostics = prognostic(state)
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_caches_result_with_datetime():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': datetime(2000, 1, 1)}
    tendencies, diagnostics = prognostic(state)
    tendencies, diagnostics = prognostic(state)
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 1


def test_set_prognostic_update_frequency_updates_result_when_equal():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic({'time': timedelta(hours=0)})
    tendencies, diagnostics = prognostic({'time': timedelta(hours=1)})
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 2


def test_set_prognostic_update_frequency_updates_result_when_greater():
    prognostic = UpdateFrequencyWrapper(MockPrognostic(), timedelta(hours=1))
    state = {'time': timedelta(hours=0)}
    tendencies, diagnostics = prognostic({'time': timedelta(hours=0)})
    tendencies, diagnostics = prognostic({'time': timedelta(hours=2)})
    assert len(diagnostics) == 1
    assert diagnostics['num_updates'] == 2


def test_put_prognostic_tendency_in_diagnostics_no_tendencies():
    class MockPrognostic(Prognostic):
        def __call__(self, state):
            return {}, {}

    prognostic = TendencyInDiagnosticsWrapper(MockPrognostic(), 'scheme')
    tendencies, diagnostics = prognostic({})
    assert len(tendencies) == 0
    assert len(diagnostics) == 0


def test_put_prognostic_tendency_in_diagnostics_one_tendency():
    class MockPrognostic(Prognostic):
        tendency_properties = {'quantity': {}}
        def __call__(self, state):
            return {'quantity': 1.}, {}

    prognostic = TendencyInDiagnosticsWrapper(MockPrognostic(), 'scheme')
    tendencies, diagnostics = prognostic({})
    assert 'tendency_of_quantity_due_to_scheme' in prognostic.diagnostics
    tendencies, diagnostics = prognostic({})
    assert 'tendency_of_quantity_due_to_scheme' in diagnostics.keys()
    assert len(diagnostics) == 1
    assert tendencies['quantity'] == 1.
    assert diagnostics['tendency_of_quantity_due_to_scheme'] == 1.
