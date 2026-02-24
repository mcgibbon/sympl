# -*- coding: utf-8 -*-
from sympl._core.combine_properties import combine_component_properties
from sympl._core.get_np_arrays import get_numpy_arrays_with_properties
from sympl._core.init_np_arrays import initialize_numpy_arrays_with_properties
from sympl._core.restore_dataarray import restore_data_arrays_with_properties

from ._components import (
    ConstantDiagnosticComponent,
    ConstantTendencyComponent,
    NetCDFMonitor,
    PlotFunctionMonitor,
    RelaxationTendencyComponent,
    RestartMonitor,
    TimeDifferencingWrapper,
)
from ._components.timesteppers import AdamsBashforth, Leapfrog, SSPRungeKutta
from ._core.backend import DataArrayBackend, StateBackend, get_backend, set_backend
from ._core.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    Monitor,
    Stepper,
    TendencyComponent,
)
from ._core.composite import (
    DiagnosticComponentComposite,
    ImplicitTendencyComponentComposite,
    MonitorComposite,
    TendencyComponentComposite,
)
from ._core.constants import (
    get_constant,
    get_constants_string,
    reset_constants,
    set_condensible_name,
    set_constant,
)
from ._core.dataarray import DataArray
from ._core.exceptions import (
    ComponentExtraOutputError,
    ComponentMissingOutputError,
    DependencyError,
    InvalidPropertyDictError,
    InvalidStateError,
    SharedKeyError,
)
from ._core.tendencystepper import TendencyStepper
from ._core.time import datetime, timedelta
from ._core.tracers import (
    get_tracer_input_properties,
    get_tracer_names,
    get_tracer_unit_dict,
    register_tracer,
)
from ._core.units import is_valid_unit, units_are_compatible, units_are_same
from ._core.util import (
    ensure_no_shared_keys,
    get_component_aliases,
    get_numpy_array,
    jit,
    restore_dimensions,
)
from ._core.wrappers import ScalingWrapper, UpdateFrequencyWrapper

__version__ = "0.4.1"
__all__ = (
    TendencyComponent,
    DiagnosticComponent,
    Stepper,
    Monitor,
    TendencyComponentComposite,
    ImplicitTendencyComponentComposite,
    DiagnosticComponentComposite,
    MonitorComposite,
    ImplicitTendencyComponent,
    TendencyStepper,
    Leapfrog,
    AdamsBashforth,
    SSPRungeKutta,
    InvalidStateError,
    SharedKeyError,
    DependencyError,
    InvalidPropertyDictError,
    ComponentExtraOutputError,
    ComponentMissingOutputError,
    units_are_same,
    units_are_compatible,
    is_valid_unit,
    DataArray,
    get_constant,
    set_constant,
    set_condensible_name,
    reset_constants,
    get_constants_string,
    TimeDifferencingWrapper,
    ensure_no_shared_keys,
    set_backend,
    get_backend,
    StateBackend,
    DataArrayBackend,
    get_numpy_array,
    jit,
    register_tracer,
    get_tracer_unit_dict,
    get_tracer_input_properties,
    get_tracer_names,
    restore_dimensions,
    get_numpy_arrays_with_properties,
    restore_data_arrays_with_properties,
    initialize_numpy_arrays_with_properties,
    get_component_aliases,
    combine_component_properties,
    PlotFunctionMonitor,
    NetCDFMonitor,
    RestartMonitor,
    ConstantTendencyComponent,
    ConstantDiagnosticComponent,
    RelaxationTendencyComponent,
    UpdateFrequencyWrapper,
    ScalingWrapper,
    datetime,
    timedelta,
)
