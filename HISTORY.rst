==========
What's New
==========

Latest
------

* Added readthedocs support
* Overhaul of documentation
* Expanded tests
* Added function to put prognostic tendencies in diagnostic output
* Class wrapping now works by inheritance, instead of by monkey patching methods

Breaking changes
~~~~~~~~~~~~~~~~

* Removed add_dicts_inplace from public API
* combine_dimensions will raise exceptions in a few more cases where it should
  do so. Particularly, if there is an extra dimension in the arrays.
* default out_dims is removed from combine_dimensions

0.1.1 (2017-01-05)
------------------

* First release on PyPI.
