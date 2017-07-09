==========
What's New
==========

Latest
------

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
