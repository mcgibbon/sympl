==========
What's New
==========

v0.4.0
------

* Stepper, DiagnosticComponent, ImplicitTendencyComponent, and TendencyComponent base classes were
  modified to include functionality that was previously in ScalingWrapper,
  UpdateFrequencyWrapper, and TendencyInDiagnosticsWrapper. The functionality of
  TendencyInDiagnosticsWrapper is now to be used in Stepper and TendencyStepper objects.
* Composites now have a component_list attribute which contains the components being
  composited.
* TimeSteppers now have a prognostic_list attribute which contains the
  prognostics used to calculate tendencies.
* TimeSteppers from sympl can now handle ImplicitTendencyComponent components.
* Added a check for netcdftime having the required objects, to fall back on not
  using netcdftime when those are missing. This is because most objects are missing in
  older versions of netcdftime (that come packaged with netCDF4) (closes #23).
* TimeSteppers should now be called with individual Prognostics as args, rather
  than a list of components, and will emit a warning when lists are given.
* TimeSteppers now have input, output, and diagnostic properties as attributes.
  These are handled entirely by the base class.
* TimeSteppers now allow you to put tendencies in their diagnostic output. This
  is done using first-order time differencing.
* Composites now have properties dictionaries.
* Updated basic components to use new component API.
* Components enforce consistency of output from array_call with properties
  dictionaries, raising ComponentMissingOutputError or ComponentExtraOutputError
  respectively if outputs do not match.
* Added a priority order of property types for determining which aliases are
  returned by get_component_aliases.
* Fixed a bug where TendencyStepper objects would modify the arrays passed to them by
  TendencyComponent objects, leading to unexpected value changes.
* Fixed a bug where constants were missing from the string returned by
  get_constants_string, particularly any new constants (issue #27)
* Fixed a bug in NetCDFMonitor which led to some aliases being skipped.
* Modified class checking on components so that components which satisfy the
  component's API will be recognized as instances using isinstance(obj, Class).
  Right now this only checks for the presence and lack of presence of
  component attributes, and correct signature of __call__. Later it may also
  check properties dictionaries for consistency, or perform other checks.
* Fixed a bug where ABCMeta was not being used in Python 3.
* Added initialize_numpy_arrays_with_properties which creates zero arrays for an output
  properties dictionary.
* Added reference_air_temperature constant.
* Fixed bug where degrees Celcius or Fahrenheit could not be used as units on inputs
  because it would lead to an error.
* Added combine_component_properties as a public function.
* Added some unit helper functions (units_are_same, units_are_compatible,
  is_valid_unit) to public API.
* Added tracer-handling funcitonality to component base classes.

Breaking changes
~~~~~~~~~~~~~~~~

* Implicit, Timestepper, Prognostic, ImplicitPrognostic, and Diagnostic objects have been renamed to
  TendencyStepper, Stepper, TendencyComponent, ImplicitTendencyComponent,
  and DiagnosticComponent. These changes are also reflected in subclass names.
* inputs, outputs, diagnostics, and tendencies are no longer attributes of components.
  In order to get these, you should use e.g. input_properties.keys()
* properties dictionaries are now abstract methods, so subclasses must define them.
  Previously they defaulted to empty dictionaries.
* Base classes now raise InvalidPropertyDictError when output property units conflict with input
  property units (which probably indicates that they're wrong).
* Components should now be written using a new array_call method rather than __call__.
  __call__ will automatically unwrap DataArrays to numpy arrays to be passed into
  array_call based on the component's properties dictionaries, and re-wrap to
  DataArrays when done.
* TimeSteppers should now be written using a _call method rather than __call__.
  __call__ wraps _call to provide some base class functionality, like putting
  tendencies in diagnostics.
* ScalingWrapper, UpdateFrequencyWrapper, and TendencyInDiagnosticsWrapper
  have been removed. The functionality of these wrappers has been moved to the
  component base types as methods and initialization options.
* 'time' now must be present in the model state dictionary. This is strictly required
  for calls to DiagnosticComponent, TendencyComponent, ImplicitTendencyComponent, and Stepper components,
  and may be strictly required in other ways in the future
* Removed everything to do with directional wildcards. Currently '*' is the
  only wildcard dimension. 'x', 'y', and 'z' refer to their own names only.
* Removed the combine_dimensions function, which wasn't used anywhere and no
  longer has much purpose without directional wildcards
* RelaxationTendencyComponent no longer allows caching of equilibrium values or
  timescale. They must be provided through the input state. This is to ensure
  proper conversion of dimensions and units.
* Removed ComponentTestBase from package. All of its tests except for output
  caching are now performed on object initialization or call time.
* "*" matches are now enforced to be the same across all quantities of a
  component, such that the length of the "*" axis will be the same for all
  quantities. Any missing dimensions that are present on other quantities
  will be created and broadcast to achieve this.
* dims_like is obsolete as a result, and is no longer used. `dims` should be
  used instead. If present, `dims` from input properties will be used as
  default.
* Components will now raise an exception when __call__ of the component base
  class (e.g. Stepper, TendencyComponent, etc.) if the __init__ method of the base
  class has not been called, telling the user that the component __init__
  method should make a call to the superclass init.

v0.3.2
------

* Exported get_constants_string to the public API
* Added "aliases" kwarg to NetCDFMonitor, allowing the monitor to shorten
  variable names when writing to netCDF
* Added get_component_aliases() to get a dictionary of quantity aliases from
  a list of Components (used by NetCDFMonitor to shorten variable
  names)
* Added tests for NetCDFMonitor aliases and get_component_aliases()

Breaking changes
~~~~~~~~~~~~~~~~
* tendencies in diagnostics are now named as X_tendency_from_Y, instead of
  tendency_of_X_due_to_Y. The idea is that it's shorter, and can easily be
  shortened more by aliasing "tendency" to "tend"

v0.3.1
------

* Fixed botched deployment, see v0.3.0 for the real changes

v0.3.0
------

* Modified component class checking to look at the presence of properties
* Added ScalingWrapper
* Fixed bug in TendencyInDiagnosticsWrapper where tendency_diagnostics_properties were
  being copied into input_properties
* Modified component class checking to look at the presence of properties
  attributes instead of checking type when verifying component class.
* Removed Python 3.4 from Travis CI testing
* added some more constants to default_constants related to conductivity of
  water in all phases and phase changes of water.
* increased the verbosity of the error output on shape mismatch in
  restore_data_arrays_with_properties
* corrected heat capacity of snow and ice to be floats instead of ints
* Added get_constant function as the way to retrieve constants
* Added ImplicitTendencyComponent as a new component type. It is like a TendencyComponent,
  but its call signature also requires that a timestep be given.
* Added TimeDifferencingWrapper, which turns an Stepper into an
  ImplicitTendencyComponent by applying first-order time differencing.
* Added set_condensible_name as a way of changing what condensible aliases
  (for example, density_of_solid_phase) refer to. Default is 'water'.
* Moved wrappers to their own file (out from util.py).
* Corrected str representation of DiagnosticComponent to say DiagnosticComponent instead of
  Stepper.
* Added a function reset_constants to reset the constants library to its
  initial state.
* Added a function datetime which accepts calendar as a keyword argument, and
  returns datetimes from netcdftime when non-default calendars are used. The
  dependency on netcdftime is optional, the other calendars just won't work if
  it isn't installed
* Added a reference to the built-in timedelta for convenience.

Breaking changes
~~~~~~~~~~~~~~~~

* Removed default_constants from the public API, use get_constant and
  set_constant instead.
* Removed replace_none_with_default. Use get_constant instead.
* set_dimension_names has been removed, use set_direction_names instead.

0.2.1
-----

* Fixed value of planetary radius, added specific heat of water vapor.
* Added function set_constant which provides an easy interface for setting
  values in the default_constants dictionary. Users can already set them
  manually by creating DataArray objects. This automates the DataArray
  creation, which should make user code cleaner.

0.2.0
-----

* Added some more physical constants.
* Added readthedocs support.
* Overhaul of documentation.
* Docstrings now use numpy style instead of Google style.
* Expanded tests.
* Added function to put prognostic tendencies in diagnostic output.
* NetCDFMonitor is actually working now, and has tests.
* There are now helper functions for automatically extracting required numpy
  arrays with correct dimensions and units from input state dictionaries. See
  the note about _properties attributes in Breaking changes below.
* Added base object for testing components
* Renamed set_dimension_names to set_direction_names, set_dimension_names is
  now deprecated and gives a warning. add_direction_names was added to append
  to the dimension list instead of replacing it.

Breaking changes
~~~~~~~~~~~~~~~~

* The constant ``stefan_boltzmann`` is now called ``stefan_boltzmann_constant``
  to maintain consistency with other names.
* Removed add_dicts_inplace from public API
* combine_dimensions will raise exceptions in a few more cases where it should
  do so. Particularly, if there is an extra dimension in the arrays.
* Default out_dims is removed from combine_dimensions.
* input_properties, tendency_properties, etc. dictionaries have been added to
  components, which contain information
  about the units and dimensions required for those arrays, and can include
  more properties as required by individual projects. This makes it possible
  to extract appropriate numpy arrays from a model state in an automated
  fashion based on these properties, significantly reducing boilerplate code.
  These dictionaries need to be defined by subclasses, instead of the old
  "inputs", "outputs" etc. lists which are auto-generated from these new
  dictionaries.
* Class wrapping now works by inheritance, instead of by monkey patching methods.
* All Exception classes (e.g. SharedKeyException) have been renamed to "Error"
  classes (e.g. SharedKeyError) to be consistent with normal Python naming
  conventions

0.1.1 (2017-01-05)
------------------

* First release on PyPI.
