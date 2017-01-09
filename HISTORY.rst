==========
What's New
==========

Latest
------

* Added readthedocs support
* Overhaul of documentation
* Expanded tests
* Added function to put prognostic tendencies in diagnostic output

Breaking changes
~~~~~~~~~~~~~~~~

* Removed add_dicts_inplace from public API
* Prognostic class no longer returned by set_prognostic_update_frequency
* combine_dimensions will raise exceptions in a few more cases where it should
  do so. Particularly, if there is an extra dimension in the arrays.
* default out_dims is removed from combine_dimensions

0.1.1 (2017-01-05)
------------------

* First release on PyPI.
