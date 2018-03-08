==========
What's New
==========

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
* Added ImplicitPrognostic as a new component type. It is like a Prognostic,
  but its call signature also requires that a timestep be given.
* Added TimeDifferencingWrapper, which turns an Implicit into an
  ImplicitPrognostic by applying first-order time differencing.
* Added set_condensible_name as a way of changing what condensible aliases
  (for example, density_of_solid_phase) refer to. Default is 'water'.
* Moved wrappers to their own file (out from util.py).
* Corrected str representation of Diagnostic to say Diagnostic instead of
  Implicit.
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
