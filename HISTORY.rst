==========
What's New
==========

Latest
------

* Added some more physical constants.
* Added readthedocs support.
* Overhaul of documentation.
* Docstrings now use numpy style instead of Google style.
* Expanded tests.
* Added function to put prognostic tendencies in diagnostic output.
* Class wrapping now works by inheritance, instead of by monkey patching methods.
* NetCDFMonitor is actually working now, and has tests.

Breaking changes
~~~~~~~~~~~~~~~~

* The constant ``stefan_boltzmann`` is now called ``stefan_boltzmann_constant`` 
  to maintain consistency with other names.
* Removed add_dicts_inplace from public API
* combine_dimensions will raise exceptions in a few more cases where it should
  do so. Particularly, if there is an extra dimension in the arrays.
* Default out_dims is removed from combine_dimensions.

0.1.1 (2017-01-05)
------------------

* First release on PyPI.
