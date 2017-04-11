import pytest
import mock
from sympl import (
    Prognostic, Diagnostic, Monitor, PrognosticComposite, DiagnosticComposite,
    MonitorComposite, SharedKeyError, DataArray
)

def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockPrognostic(Prognostic):

    def __call__(self, state):
        return {}, {}


class MockPrognostic2(Prognostic):

    def __call__(self, state):
        return {}, {}


class MockDiagnostic(Diagnostic):

    def __call__(self, state):
        return {}


class MockMonitor(Monitor):

    def store(self, state):
        return


def test_empty_prognostic_composite():
    prognostic_composite = PrognosticComposite()
    state = {'air_temperature': 273.15}
    tendencies, diagnostics = prognostic_composite(state)
    assert len(tendencies) == 0
    assert len(diagnostics) == 0
    assert isinstance(tendencies, dict)
    assert isinstance(diagnostics, dict)


@mock.patch.object(MockPrognostic, '__call__')
def test_prognostic_composite_calls_one_prognostic(mock_call):
    mock_call.return_value = ({'air_temperature': 0.5}, {'foo': 50.})
    prognostic_composite = PrognosticComposite(MockPrognostic())
    state = {'air_temperature': 273.15}
    tendencies, diagnostics = prognostic_composite(state)
    assert mock_call.called
    assert tendencies == {'air_temperature': 0.5}
    assert diagnostics == {'foo': 50.}


@mock.patch.object(MockPrognostic, '__call__')
def test_prognostic_composite_calls_two_prognostics(mock_call):
    mock_call.return_value = ({'air_temperature': 0.5}, {})
    prognostic_composite = PrognosticComposite(
        MockPrognostic(), MockPrognostic())
    state = {'air_temperature': 273.15}
    tendencies, diagnostics = prognostic_composite(state)
    assert mock_call.called
    assert mock_call.call_count == 2
    assert tendencies == {'air_temperature': 1.}
    assert diagnostics == {}


def test_empty_diagnostic_composite():
    diagnostic_composite = DiagnosticComposite()
    state = {'air_temperature': 273.15}
    diagnostics = diagnostic_composite(state)
    assert len(diagnostics) == 0
    assert isinstance(diagnostics, dict)


@mock.patch.object(MockDiagnostic, '__call__')
def test_diagnostic_composite_calls_one_diagnostic(mock_call):
    mock_call.return_value = {'foo': 50.}
    diagnostic_composite = DiagnosticComposite(MockDiagnostic())
    state = {'air_temperature': 273.15}
    diagnostics = diagnostic_composite(state)
    assert mock_call.called
    assert diagnostics == {'foo': 50.}


def test_empty_monitor_collection():
    # mainly we're testing that nothing errors
    monitor_collection = MonitorComposite()
    state = {'air_temperature': 273.15}
    monitor_collection.store(state)


@mock.patch.object(MockMonitor, 'store')
def test_monitor_collection_calls_one_monitor(mock_store):
    mock_store.return_value = None
    monitor_collection = MonitorComposite(MockMonitor())
    state = {'air_temperature': 273.15}
    monitor_collection.store(state)
    assert mock_store.called


@mock.patch.object(MockMonitor, 'store')
def test_monitor_collection_calls_two_monitors(mock_store):
    mock_store.return_value = None
    monitor_collection = MonitorComposite(MockMonitor(), MockMonitor())
    state = {'air_temperature': 273.15}
    monitor_collection.store(state)
    assert mock_store.called
    assert mock_store.call_count == 2


def test_prognostic_composite_cannot_use_diagnostic():
    try:
        PrognosticComposite(MockDiagnostic())
    except TypeError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('TypeError should have been raised')


def test_diagnostic_composite_cannot_use_prognostic():
    try:
        DiagnosticComposite(MockPrognostic())
    except TypeError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('TypeError should have been raised')


@mock.patch.object(MockDiagnostic, '__call__')
def test_diagnostic_composite_call(mock_call):
    mock_call.return_value = {'foo': 5.}
    state = {'bar': 10.}
    diagnostics = DiagnosticComposite(MockDiagnostic())
    new_state = diagnostics(state)
    assert list(state.keys()) == ['bar']
    assert state['bar'] == 10.
    assert list(new_state.keys()) == ['foo']
    assert new_state['foo'] == 5.


def test_prognostic_composite_includes_attributes():
    prognostic = MockPrognostic()
    prognostic.input_properties = {'input1': {'units': 'K', 'dims':['z', 'y', 'x']}}
    prognostic.diagnostic_properties = {'diagnostic1': {'units': 'm/s'}}
    prognostic.tendency_properties = {'tendency1': {}}
    composite = PrognosticComposite(prognostic)
    assert composite.inputs == ('input1',)
    assert composite.diagnostics == ('diagnostic1',)
    assert composite.tendencies == ('tendency1',)


def test_prognostic_composite_includes_attributes_from_two():
    prognostic1 = MockPrognostic()
    prognostic1.input_properties = {'input1': {'units': 'K', 'dims':['z', 'y', 'x']}}
    prognostic1.diagnostic_properties = {'diagnostic1': {'units': 'm/s'}}
    prognostic1.tendency_properties = {'tendency1': {}}
    prognostic2 = MockPrognostic()
    prognostic2.input_properties = {'input2': {'units': 'K', 'dims':['z', 'y', 'x']}}
    prognostic2.diagnostic_properties = {'diagnostic2': {'units': 'm/s'}}
    prognostic2.tendency_properties = {'tendency2': {}}
    composite = PrognosticComposite(prognostic1, prognostic2)
    assert same_list(composite.inputs, ('input1', 'input2'))
    assert same_list(composite.diagnostics, ('diagnostic1', 'diagnostic2'))
    assert same_list(composite.tendencies, ('tendency1', 'tendency2'))


def test_prognostic_merges_attributes():
    prognostic1 = MockPrognostic()
    prognostic1.input_properties = {'input1': {}}
    prognostic1.diagnostic_properties = {'diagnostic1': {}}
    prognostic1.tendency_properties = {'tendency1': {}, 'tendency2': {}}
    prognostic2 = MockPrognostic()
    prognostic2.input_properties = {'input1': {}, 'input2': {}}
    prognostic2.diagnostic_properties = {'diagnostic2': {}}
    prognostic2.tendency_properties = {'tendency2': {}}
    composite = PrognosticComposite(prognostic1, prognostic2)
    assert same_list(composite.inputs, ('input1', 'input2'))
    assert same_list(composite.diagnostics, ('diagnostic1', 'diagnostic2'))
    assert same_list(composite.tendencies, ('tendency1', 'tendency2'))


def test_prognostic_composite_ensures_valid_state():
    prognostic1 = MockPrognostic()
    prognostic1.input_properties = {'input1': {}}
    prognostic1.diagnostic_properties = {'diagnostic1': {}}
    prognostic2 = MockPrognostic()
    prognostic2.input_properties = {'input1': {}, 'input2': {}}
    prognostic2.diagnostic_properties = {'diagnostic1': {}}
    try:
        PrognosticComposite(prognostic1, prognostic2)
    except SharedKeyError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError(
            'Should not be able to have overlapping diagnostics in composite')


@mock.patch.object(MockPrognostic, '__call__')
@mock.patch.object(MockPrognostic2, '__call__')
def test_prognostic_component_handles_units_when_combining(mock_call, mock2_call):
    mock_call.return_value = ({
        'eastward_wind': DataArray(1., attrs={'units': 'm/s'})}, {})
    mock2_call.return_value = ({
        'eastward_wind': DataArray(50., attrs={'units': 'cm/s'})}, {})
    prognostic1 = MockPrognostic()
    prognostic2 = MockPrognostic2()
    composite = PrognosticComposite(prognostic1, prognostic2)
    tendencies, diagnostics = composite({})
    assert tendencies['eastward_wind'].to_units('m/s').values.item() == 1.5


def test_diagnostic_composite_includes_attributes():
    diagnostic = MockDiagnostic()
    diagnostic.input_properties = {'input1': {}}
    diagnostic.diagnostic_properties = {'diagnostic1': {}}
    composite = DiagnosticComposite(diagnostic)
    assert composite.inputs == ('input1',)
    assert composite.diagnostics == ('diagnostic1',)


def test_diagnostic_composite_includes_attributes_from_two():
    diagnostic1 = MockDiagnostic()
    diagnostic1.input_properties = {'input1': {}}
    diagnostic1.diagnostic_properties = {'diagnostic1': {}}
    diagnostic2 = MockDiagnostic()
    diagnostic2.input_properties = {'input2': {}}
    diagnostic2.diagnostic_properties = {'diagnostic2': {}}
    composite = DiagnosticComposite(diagnostic1, diagnostic2)
    assert same_list(composite.inputs, ('input1', 'input2'))
    assert same_list(composite.diagnostics, ('diagnostic1', 'diagnostic2'))


def test_diagnostic_composite_merges_attributes():
    diagnostic1 = MockDiagnostic()
    diagnostic1.input_properties = {'input1': {}}
    diagnostic1.diagnostic_properties = {'diagnostic1': {}}
    diagnostic2 = MockDiagnostic()
    diagnostic2.input_properties = {'input1': {}, 'input2': {}}
    diagnostic2.diagnostic_properties = {'diagnostic2': {}}
    composite = DiagnosticComposite(diagnostic1, diagnostic2)
    assert same_list(composite.inputs, ('input1', 'input2'))
    assert same_list(composite.diagnostics, ('diagnostic1', 'diagnostic2'))


def test_diagnostic_composite_ensures_valid_state():
    diagnostic1 = MockDiagnostic()
    diagnostic1.input_properties = {'input1': {}}
    diagnostic1.diagnostic_properties = {'diagnostic1': {}}
    diagnostic2 = MockDiagnostic()
    diagnostic2.input_properties = {'input1': {}, 'input2': {}}
    diagnostic2.diagnostic_properties = {'diagnostic1': {}}
    try:
        DiagnosticComposite(diagnostic1, diagnostic2)
    except SharedKeyError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError(
            'Should not be able to have overlapping diagnostics in composite')


if __name__ == '__main__':
    pytest.main([__file__])
