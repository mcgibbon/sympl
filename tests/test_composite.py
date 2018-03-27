import pytest
import unittest
import mock
from sympl import (
    Prognostic, Diagnostic, Monitor, PrognosticComposite, DiagnosticComposite,
    MonitorComposite, SharedKeyError, DataArray, InvalidPropertyDictError
)
from sympl._core.units import units_are_compatible


def same_list(list1, list2):
    return (len(list1) == len(list2) and all(
        [item in list2 for item in list1] + [item in list1 for item in list2]))


class MockPrognostic(Prognostic):

    input_properties = None
    diagnostic_properties = None
    tendency_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, tendency_properties,
            diagnostic_output, tendency_output, **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self.tendency_properties = tendency_properties
        self._diagnostic_output = diagnostic_output
        self._tendency_output = tendency_output
        self.times_called = 0
        self.state_given = None
        super(MockPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self._tendency_output, self._diagnostic_output


class MockDiagnostic(Diagnostic):

    input_properties = None
    diagnostic_properties = None

    def __init__(
            self, input_properties, diagnostic_properties, diagnostic_output,
            **kwargs):
        self.input_properties = input_properties
        self.diagnostic_properties = diagnostic_properties
        self._diagnostic_output = diagnostic_output
        self.times_called = 0
        self.state_given = None
        super(MockDiagnostic, self).__init__(**kwargs)

    def array_call(self, state):
        self.times_called += 1
        self.state_given = state
        return self._diagnostic_output


class MockEmptyPrognostic(Prognostic):

    input_properties = {}
    diagnostic_properties = {}
    tendency_properties = {}

    def __init__(self, **kwargs):
        super(MockEmptyPrognostic, self).__init__(**kwargs)

    def array_call(self, state):
        return {}, {}


class MockEmptyPrognostic2(Prognostic):

    input_properties = {}
    diagnostic_properties = {}
    tendency_properties = {}

    def __init__(self, **kwargs):
        super(MockEmptyPrognostic2, self).__init__(**kwargs)

    def array_call(self, state):
        return {}, {}


class MockEmptyDiagnostic(Diagnostic):

    input_properties = {}
    diagnostic_properties = {}

    def __init__(self, **kwargs):
        super(MockEmptyDiagnostic, self).__init__(**kwargs)

    def array_call(self, state):
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


@mock.patch.object(MockEmptyPrognostic, '__call__')
def test_prognostic_composite_calls_one_prognostic(mock_call):
    mock_call.return_value = ({'air_temperature': 0.5}, {'foo': 50.})
    prognostic_composite = PrognosticComposite(MockEmptyPrognostic())
    state = {'air_temperature': 273.15}
    tendencies, diagnostics = prognostic_composite(state)
    assert mock_call.called
    assert tendencies == {'air_temperature': 0.5}
    assert diagnostics == {'foo': 50.}


@mock.patch.object(MockEmptyPrognostic, '__call__')
def test_prognostic_composite_calls_two_prognostics(mock_call):
    mock_call.return_value = ({'air_temperature': 0.5}, {})
    prognostic_composite = PrognosticComposite(
        MockEmptyPrognostic(), MockEmptyPrognostic())
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


@mock.patch.object(MockEmptyDiagnostic, '__call__')
def test_diagnostic_composite_calls_one_diagnostic(mock_call):
    mock_call.return_value = {'foo': 50.}
    diagnostic_composite = DiagnosticComposite(MockEmptyDiagnostic())
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
        PrognosticComposite(MockEmptyDiagnostic())
    except TypeError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('TypeError should have been raised')


def test_diagnostic_composite_cannot_use_prognostic():
    try:
        DiagnosticComposite(MockEmptyPrognostic())
    except TypeError:
        pass
    except Exception as err:
        raise err
    else:
        raise AssertionError('TypeError should have been raised')


@mock.patch.object(MockEmptyDiagnostic, '__call__')
def test_diagnostic_composite_call(mock_call):
    mock_call.return_value = {'foo': 5.}
    state = {'bar': 10.}
    diagnostics = DiagnosticComposite(MockEmptyDiagnostic())
    new_state = diagnostics(state)
    assert list(state.keys()) == ['bar']
    assert state['bar'] == 10.
    assert list(new_state.keys()) == ['foo']
    assert new_state['foo'] == 5.


@mock.patch.object(MockEmptyPrognostic, '__call__')
@mock.patch.object(MockEmptyPrognostic2, '__call__')
def test_prognostic_component_handles_units_when_combining(mock_call, mock2_call):
    mock_call.return_value = ({
        'eastward_wind': DataArray(1., attrs={'units': 'm/s'})}, {})
    mock2_call.return_value = ({
        'eastward_wind': DataArray(50., attrs={'units': 'cm/s'})}, {})
    prognostic1 = MockEmptyPrognostic()
    prognostic2 = MockEmptyPrognostic2()
    composite = PrognosticComposite(prognostic1, prognostic2)
    tendencies, diagnostics = composite({})
    assert tendencies['eastward_wind'].to_units('m/s').values.item() == 1.5


def test_diagnostic_composite_single_component_input():
    input_properties = {
        'input1': {
            'dims': ['dim1'],
            'units': 'm',
        },
        'input2': {
            'dims': ['dim2'],
            'units': 'm/s'
        },
    }
    diagnostic_properties = {}
    diagnostic_output = {}
    diagnostic = MockDiagnostic(
        input_properties, diagnostic_properties, diagnostic_output)
    composite = DiagnosticComposite(diagnostic)
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_single_component_diagnostic():
    input_properties = {}
    diagnostic_properties = {
        'diag1': {
            'dims': ['lon'],
            'units': 'km',
        },
        'diag2': {
            'dims': ['lon'],
            'units': 'degK',
        },
    }
    diagnostic_output = {}
    diagnostic = MockDiagnostic(
        input_properties, diagnostic_properties, diagnostic_output)
    composite = DiagnosticComposite(diagnostic)
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_single_empty_component():
    input_properties = {}
    diagnostic_properties = {}
    diagnostic_output = {}
    diagnostic = MockDiagnostic(
        input_properties, diagnostic_properties, diagnostic_output)
    composite = DiagnosticComposite(diagnostic)
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_single_full_component():
    input_properties = {
        'input1': {
            'dims': ['dim1'],
            'units': 'm',
        },
        'input2': {
            'dims': ['dim2'],
            'units': 'm/s'
        },
    }
    diagnostic_properties = {
        'diag1': {
            'dims': ['lon'],
            'units': 'km',
        },
        'diag2': {
            'dims': ['lon'],
            'units': 'degK',
        },
    }
    diagnostic_output = {}
    diagnostic = MockDiagnostic(
        input_properties, diagnostic_properties, diagnostic_output)
    composite = DiagnosticComposite(diagnostic)
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_single_component_no_dims_on_diagnostic():
    input_properties = {
        'diag1': {
            'dims': ['dim1'],
            'units': 'm',
        },
    }
    diagnostic_properties = {
        'diag1': {
            'units': 'km',
        },
    }
    diagnostic_output = {}
    diagnostic = MockDiagnostic(
        input_properties, diagnostic_properties, diagnostic_output)
    composite = DiagnosticComposite(diagnostic)
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_single_component_missing_dims_on_diagnostic():
    input_properties = {}
    diagnostic_properties = {
        'diag1': {
            'units': 'km',
        },
    }
    diagnostic_output = {}
    try:
        diagnostic = MockDiagnostic(
            input_properties, diagnostic_properties, diagnostic_output)
        DiagnosticComposite(diagnostic)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_diagnostic_composite_two_components_no_overlap():
    diagnostic1 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
        },
        diagnostic_properties={
            'diag1': {
                'dims': ['lon'],
                'units': 'km',
            },
        },
        diagnostic_output={}
    )
    diagnostic2 = MockDiagnostic(
        input_properties={
            'input2': {
                'dims': ['dim2'],
                'units': 'm/s'
            },
        },
        diagnostic_properties={
            'diag2': {
                'dims': ['lon'],
                'units': 'degK',
            },
        },
        diagnostic_output={}
    )
    composite = DiagnosticComposite(diagnostic1, diagnostic2)
    input_properties = {
        'input1': {
            'dims': ['dim1'],
            'units': 'm',
        },
        'input2': {
            'dims': ['dim2'],
            'units': 'm/s'
        },
    }
    diagnostic_properties = {
        'diag1': {
            'dims': ['lon'],
            'units': 'km',
        },
        'diag2': {
            'dims': ['lon'],
            'units': 'degK',
        },
    }
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_two_components_overlap_input():
    diagnostic1 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
            'input2': {
                'dims': ['dim2'],
                'units': 'm/s'
            },
        },
        diagnostic_properties={
            'diag1': {
                'dims': ['lon'],
                'units': 'km',
            },
        },
        diagnostic_output={}
    )
    diagnostic2 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            },
            'input2': {
                'dims': ['dim2'],
                'units': 'm/s'
            },
        },
        diagnostic_properties={
            'diag2': {
                'dims': ['lon'],
                'units': 'degK',
            },
        },
        diagnostic_output={}
    )
    composite = DiagnosticComposite(diagnostic1, diagnostic2)
    input_properties = {
        'input1': {
            'dims': ['dim1'],
            'units': 'm',
        },
        'input2': {
            'dims': ['dim2'],
            'units': 'm/s'
        },
    }
    diagnostic_properties = {
        'diag1': {
            'dims': ['lon'],
            'units': 'km',
        },
        'diag2': {
            'dims': ['lon'],
            'units': 'degK',
        },
    }
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties


def test_diagnostic_composite_two_components_overlap_diagnostic():
    diagnostic1 = MockDiagnostic(
        input_properties={},
        diagnostic_properties={
            'diag1': {
                'dims': ['lon'],
                'units': 'km',
            },
        },
        diagnostic_output={}
    )
    diagnostic2 = MockDiagnostic(
        input_properties={},
        diagnostic_properties={
            'diag1': {
                'dims': ['lon'],
                'units': 'km',
            },
        },
        diagnostic_output={}
    )
    try:
        DiagnosticComposite(diagnostic1, diagnostic2)
    except SharedKeyError:
        pass
    else:
        raise AssertionError('Should have raised SharedKeyError')


def test_diagnostic_composite_two_components_incompatible_input_dims():
    diagnostic1 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        },
        diagnostic_properties={},
        diagnostic_output={}
    )
    diagnostic2 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim2'],
                'units': 'm',
            }
        },
        diagnostic_properties={},
        diagnostic_output={}
    )
    try:
        composite = DiagnosticComposite(diagnostic1, diagnostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_diagnostic_composite_two_components_incompatible_input_units():
    diagnostic1 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        },
        diagnostic_properties={},
        diagnostic_output={}
    )
    diagnostic2 = MockDiagnostic(
        input_properties={
            'input1': {
                'dims': ['dim1'],
                'units': 's',
            }
        },
        diagnostic_properties={},
        diagnostic_output={}
    )
    try:
        DiagnosticComposite(diagnostic1, diagnostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_prognostic_composite_single_input():
    prognostic = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1'],
                'units': 'm',
            }
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_prognostic_composite_single_diagnostic():
    prognostic = MockPrognostic(
        input_properties={},
        diagnostic_properties={
            'diag1': {
                'dims': ['dims2'],
                'units': '',
            }
        },
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_prognostic_composite_single_tendency():
    prognostic = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_prognostic_composite_implicit_dims():
    prognostic = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == {
        'tend1': {
            'dims': ['dims1', 'dims2'],
            'units': 'degK / day',
        }
    }


def test_two_prognostic_composite_implicit_dims():
    prognostic = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic, prognostic2)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == {
        'tend1': {
            'dims': ['dims1', 'dims2'],
            'units': 'degK / day',
        }
    }


def test_prognostic_composite_explicit_dims():
    prognostic = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_two_prognostic_composite_explicit_dims():
    prognostic = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic, prognostic2)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_two_prognostic_composite_explicit_and_implicit_dims():
    prognostic = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic, prognostic2)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_prognostic_composite_explicit_dims_not_in_input():
    prognostic = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_two_prognostic_composite_incompatible_dims():
    prognostic = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            },
            'input2': {
                'dims': ['dims3', 'dims1'],
                'units': 'degK / day'
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            },
            'input2': {
                'dims': ['dims3', 'dims1'],
                'units': 'degK / day'
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims3', 'dims1'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    try:
        PrognosticComposite(prognostic, prognostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_two_prognostic_composite_compatible_dims():
    prognostic = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            },
            'input2': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day'
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            },
            'input2': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day'
            }
        },
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK / day',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic, prognostic2)
    assert composite.input_properties == prognostic.input_properties
    assert composite.diagnostic_properties == prognostic.diagnostic_properties
    assert composite.tendency_properties == prognostic.tendency_properties


def test_prognostic_composite_two_components_input():
    prognostic1 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK',
            },
            'input2': {
                'dims': ['dims1'],
                'units': 'm',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK',
            },
            'input3': {
                'dims': ['dims2'],
                'units': '',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic1, prognostic2)
    input_properties = {
        'input1': {
            'dims': ['dims1', 'dims2'],
            'units': 'degK',
        },
        'input2': {
            'dims': ['dims1'],
            'units': 'm',
        },
        'input3': {
            'dims': ['dims2'],
            'units': '',
        },
    }
    diagnostic_properties = {}
    tendency_properties = {}
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties
    assert composite.tendency_properties == tendency_properties


def test_prognostic_composite_two_components_swapped_input_dims():
    prognostic1 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims2', 'dims1'],
                'units': 'degK',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic1, prognostic2)
    diagnostic_properties = {}
    tendency_properties = {}
    assert (composite.input_properties == prognostic1.input_properties or
            composite.input_properties == prognostic2.input_properties)
    assert composite.diagnostic_properties == diagnostic_properties
    assert composite.tendency_properties == tendency_properties


def test_prognostic_composite_two_components_incompatible_input_dims():
    prognostic1 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims3'],
                'units': 'degK',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    try:
        PrognosticComposite(prognostic1, prognostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_prognostic_composite_two_components_incompatible_input_units():
    prognostic1 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'degK',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'm',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    try:
        PrognosticComposite(prognostic1, prognostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_prognostic_composite_two_components_compatible_input_units():
    prognostic1 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'km',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={
            'input1': {
                'dims': ['dims1', 'dims2'],
                'units': 'cm',
            },
        },
        diagnostic_properties={},
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic1, prognostic2)
    assert 'input1' in composite.input_properties.keys()
    assert composite.input_properties['input1']['dims'] == ['dims1', 'dims2']
    assert units_are_compatible(composite.input_properties['input1']['units'], 'm')


def test_prognostic_composite_two_components_tendency():
    prognostic1 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/s'
            },
            'tend2': {
                'dims': ['dim1'],
                'units': 'degK/s'
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic1, prognostic2)
    input_properties = {}
    diagnostic_properties = {}
    tendency_properties = {
        'tend1': {
            'dims': ['dim1'],
            'units': 'm/s'
        },
        'tend2': {
            'dims': ['dim1'],
            'units': 'degK/s'
        }
    }
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties
    assert composite.tendency_properties == tendency_properties


def test_prognostic_composite_two_components_tendency_incompatible_dims():
    prognostic1 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim2'],
                'units': 'm/s'
            },
            'tend2': {
                'dims': ['dim1'],
                'units': 'degK/s'
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    try:
        PrognosticComposite(prognostic1, prognostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_prognostic_composite_two_components_tendency_incompatible_units():
    prognostic1 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/s',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'degK/s'
            },
            'tend2': {
                'dims': ['dim1'],
                'units': 'degK/s'
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    try:
        PrognosticComposite(prognostic1, prognostic2)
    except InvalidPropertyDictError:
        pass
    else:
        raise AssertionError('Should have raised InvalidPropertyDictError')


def test_prognostic_composite_two_components_tendency_compatible_units():
    prognostic1 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'km/s',
            }
        },
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={},
        diagnostic_properties={},
        tendency_properties={
            'tend1': {
                'dims': ['dim1'],
                'units': 'm/day'
            },
        },
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic1, prognostic2)
    assert 'tend1' in composite.tendency_properties.keys()
    assert composite.tendency_properties['tend1']['dims'] == ['dim1']
    assert units_are_compatible(composite.tendency_properties['tend1']['units'], 'm/s')


def test_prognostic_composite_two_components_diagnostic():
    prognostic1 = MockPrognostic(
        input_properties={},
        diagnostic_properties={
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        },
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={},
        diagnostic_properties={
            'diag2': {
                'dims': ['dim2'],
                'units': 'm',
            }
        },
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    composite = PrognosticComposite(prognostic1, prognostic2)
    input_properties = {}
    diagnostic_properties = {
        'diag1': {
            'dims': ['dim1'],
            'units': 'm',
        },
        'diag2': {
            'dims': ['dim2'],
            'units': 'm',
        },
    }
    tendency_properties = {}
    assert composite.input_properties == input_properties
    assert composite.diagnostic_properties == diagnostic_properties
    assert composite.tendency_properties == tendency_properties


def test_prognostic_composite_two_components_overlapping_diagnostic():
    prognostic1 = MockPrognostic(
        input_properties={},
        diagnostic_properties={
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        },
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    prognostic2 = MockPrognostic(
        input_properties={},
        diagnostic_properties={
            'diag1': {
                'dims': ['dim1'],
                'units': 'm',
            }
        },
        tendency_properties={},
        diagnostic_output={},
        tendency_output={},
    )
    try:
        PrognosticComposite(prognostic1, prognostic2)
    except SharedKeyError:
        pass
    else:
        raise AssertionError('Should have raised SharedKeyError')


if __name__ == '__main__':
    pytest.main([__file__])
