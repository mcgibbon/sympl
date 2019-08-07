====================
Overview: Why Sympl?
====================

Traditional atmospheric and Earth system models can be hard to read and change
for many reasons. Sympl tries to learn from the past experience of these models
to speed up research and improve accessibility.

At its core, Sympl defines a model in terms of a `state` that gets changed in
sequence by components of a model (like the radiation scheme, or dynamical
core). Each of those components as well-defined and documented inputs and
outputs, and code in Sympl will automatically handle unit and dimensionality
conversions (such as dimension orderings) to give components the inputs they
need.

Sympl defines a framework of Python object interfaces (APIs) that can be
combined to create a model. This has many benefits:

* Objects can use code written in any language that can be called from Python,
  including Fortran, C, C++, Julia, Matlab, and others.
* Each object, such as a radiation parameterization, has a clearly documented
  interface and can be understood without looking at any other part of a
  model's code. Certain interfaces have been designed to force model code to
  self-document, such as having inputs and outputs as properties of a scheme.
* Objects can be swapped out with other compatible objects. For example, Sympl
  makes it trivial to change the type of time stepping used on a prognostic
  scheme.
* Code can be re-used between different types of models. For instance, an
  atmospheric general circulation model, numerical weather prediction model,
  and large-eddy simulation could all use the same RRTM radiation object.
* Already-existing documentation for Sympl can tell your users how to configure
  and run your model. You will likely spend less time writing documentation,
  but end up with a better documented model. As long as you write docstrings,
  you're good to go!

Sympl also contains a number of commonly used objects, such
as time steppers and NetCDF output objects.

So is Sympl a model?
--------------------

Sympl is *not* a model itself. In particular, physical parameterizations and
dynamical cores are not present in Sympl. This code instead can be found in
other projects that make use of Sympl.

Sympl is meant to be a community ecosystem that allows researchers and other
users to use and combine components from a number of different sources.
By keeping model physics/dynamics code outside of Sympl itself, researchers
can own and maintain their own models. The framework API ensures that models
using Sympl are clear and accessible, and allows components from different models
and packages to be used alongside one another.

Then where's the model?
-----------------------

Check out `CliMT <https://github.com/climt/climt>`_ as an example. We highly
recommend reading the `paper on Sympl and CliMT`_.

In Sympl, the "model" in the traditional sense of Fortran models is essentially
your run script. You use a Python run script instead of a Fortran module like
`main.f90`. This may sound scary, but the idea is that the python run script
is as easy (or easier) to understand than a configuration file. Sympl makes
choices that force those run scripts to be easier to understand.
You can see examples in the above paper.

A "model developer" in the traditional sense would write a toolkit package
containing model components that are used by those run scripts. See CliMT for
an example of this. That toolkit should also come with example run scripts
using its components.

In a way, when you configure the model you are writing the model itself. This
is reasonable in Sympl because the model run script should be accessible and
readable by users with basic knowledge of programming (even users who don't
know Python). By being readable, the model run script tells others clearly and
precisely how you configured and ran your model.

If someone wants to write a model in the traditional way, where their Python
run script is never changed and instead you configure the model by changing
a configuration file, they can do that, too! Read about the particular model
you're using for details.

The API
-------

In a Sympl model, the model
state is contained within a "state dictionary". This is a Python dictionary
whose keys are strings indicating a quantity, and values are DataArrays with
the values of those quantities. The one exception is "time", which is stored
as a timedelta or datetime-like object, not as a DataArray. The DataArrays
also contain information about the units of the quantity, and the grid it is
located on. At the start of a model script, the state dictionary should be
set to initial values. Code to do this may be present in other packages, or you
can write this code yourself. The state and its initialization is discussed
further in :ref:`Model State`.

The state dictionary is evolved by :py:class:`~sympl.TendencyStepper` and
:py:class:`~sympl.Stepper` objects. These types of objects take in the state
and a timedelta object that indicates the time step, and return the next
model state. :py:class:`~sympl.TendencyStepper` objects do this by wrapping
:py:class:`~sympl.TendencyComponent` objects, which calculate tendencies using the
state dictionary. We should note that the meaning of "Stepper" in Sympl is
slightly different than its traditional definition. Here an "Stepper" object is
one that calculates the new state directly from the current state, or any
object that requires the timestep to calculate the new state, while
"TendencyComponent" objects are ones that calculate tendencies without using the
timestep. If a :py:class:`~sympl.TendencyStepper` or :py:class:`~sympl.Stepper`
object needs to use multiple time steps in its calculation, it does so by
storing states it was previously given until they are no longer needed.

The state is also calculated using :py:class:`~sympl.DiagnosticComponent` objects which
determine diagnostic quantities at the current time from the current state,
returning them in a new dictionary. This type of object is particularly useful
if you want to write your own online diagnostics.

The state can be stored or viewed using :py:class:`~sympl.Monitor` objects.
These take in the model state and do something with it, such as storing it in
a NetCDF file, or updating an interactive plot that is being shown to the user.

.. _paper on Sympl and CliMT:  https://www.geosci-model-dev.net/11/3781/2018/
