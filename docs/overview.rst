====================
Overview: Why Sympl?
====================

Traditional atmospheric and Earth system models can be difficult to understand
and modify for a number of reasons. Sympl aims to learn from the past
experience of these models to accelerate research and improve accessibility.

Sympl defines a set of Python object APIs that can be combined to create a
model. This has a number of benefits:

* Objects can use code written in any language that can be called from Python,
  including Fortran, C, C++, Julia, Matlab, and others.
* Each object, such as a radiation parameterization, has a clearly documented
  interface and can be understood without looking at any other part of a
  model's code. Certain interfaces have been designed to force model code to
  self-document, such as having inputs and outputs as properties of a scheme.
* Objects can be swapped out with other compatible objects. For example, Sympl
  makes it trivial to change the type of time stepping used.
* Code can be re-used between different types of models. For instance, an
  atmospheric general circulation model, numerical weather prediction model,
  and large-eddy simulation could all use the same RRTM radiation object.

Sympl is also a toolkit which contains a number of commonly used objects, such
as time steppers and NetCDF output objects.

Sympl as a community
--------------------

Sympl is *not* a model itself. In particular, physical parameterizations and
dynamical cores are not present in Sympl. This code instead can be found in
other projects that make use of Sympl.

Sympl is meant to be a community ecosystem that allows researchers and other
users to use and combine components from a number of different sources.
By keeping model physics/dynamics code outside of Sympl itself, researchers
can own and maintain their own models. The framework API ensures that models
using Sympl are clear and accessible, and that components from different models
and packages are compatible with one another.
