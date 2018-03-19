==========
Quickstart
==========

Here we have an example of how Sympl might be used to construct a model run
script, with explanations of what's going on. Here is the full model script we
will be looking at:

.. code-block:: python

    from model_package import (
        get_initial_state, Radiation, BoundaryLayer, DeepConvection,
        ImplicitDynamics)
    from sympl import (
        AdamsBashforth, PlotFunctionMonitor, UpdateFrequencyWrapper,
        datetime, timedelta)

    def my_plot_function(fig, state):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_title('Lowest model level air temperature (K)')
        im = ax.pcolormesh(
            state['air_temperature'].to_units('degK').values[0, :, :],
            vmin=260.,
            vmax=310.)
        cbar = fig.colorbar(im)

    plot_monitor = PlotFunctionMonitor(my_plot_function)

    state = get_initial_state(nx=256, ny=128, nz=64)
    state['time'] = datetime(2000, 1, 1)

    physics_stepper = AdamsBashforth([
        UpdateFrequencyWrapper(Radiation(), timedelta(hours=2)),
        BoundaryLayer(),
        DeepConvection(),
    ])
    implicit_dynamics = ImplicitDynamics()

    timestep = timedelta(minutes=30)
    while state['time'] < datetime(2010, 1, 1):
        physics_diagnostics, state_after_physics = physics_stepper(state, timestep)
        dynamics_diagnostics, next_state = implicit_dynamics(state_after_physics, timestep)
        state.update(physics_diagnostics)
        state.update(dynamics_diagnostics)
        plot_monitor.store(state)
        next_state['time'] = state['time'] + timestep
        state = next_state

Importing Packages
------------------

At the beginning of the script we have import statements:

.. code-block:: python

    from model_package import (
        get_initial_state, Radiation, BoundaryLayer, DeepConvection,
        ImplicitDynamics)
    from sympl import (
        AdamsBashforth, PlotFunctionMonitor, UpdateFrequencyWrapper,
        datetime, timedelta)

These grant access to the objects that will be used to construct the model,
and are dependent on the model package you are using. Here, the names
:py:class:`model_package`, :py:class:`get_initial_state`, :py:class:`Radiation`,
:py:class:`BoundaryLayer`, :py:class:`DeepConvection`, and
:py:class:`ImplicitDynamics` are placeholders, and do not refer to
an actual existing package.

Defining a PlotFunctionMonitor
------------------------------

Here we define a plotting function, and use it to create a
:py:class:`~sympl.Monitor` using :py:class:`~sympl.PlotFunctionMonitor`:

.. code-block:: python

    def my_plot_function(fig, state):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_title('Lowest model level air temperature (K)')
        im = ax.pcolormesh(
            state['air_temperature'].to_units('degK').values[0, :, :],
            vmin=260.,
            vmax=310.)
        cbar = fig.colorbar(im)

    plot_monitor = PlotFunctionMonitor(my_plot_function)

That :py:class:`~sympl.Monitor` will be used to produce an animated plot of the lowest model
level air temperature as the model runs. Here we assume that the first axis
is the vertical axis, and that the lowest level is at the lowest index, but
this depends entirely on your model. The ``[0, :, :]`` part might be different
for your model.

Initialize the Model State
--------------------------

To initialize the model, we need to create a dictionary which contains the
model state. The way this is done is model-dependent. Here we assume there is
a function that was defined by the `model_package` package which handles this
for us:

.. code-block:: python

    state = get_initial_state(nx=256, ny=128, nz=64)
    state['time'] = datetime(2000, 1, 1)

An initialized `state` is a dictionary whose keys are strings (like
'air_temperature') and values are :py:class:`~sympl.DataArray` objects, which
store not only the data but also metadata like units. The one exception
is the "time" quantity which is either a `datetime`-like or `timedelta`-like
object. Here we are calling :py:func:`sympl.datetime` to initialize time,
rather than directly creating a Python datetime. This is because
:py:func:`sympl.datetime` can support a number of calendars using the
`netcdftime` package, if installed, unlike the built-in `datetime` which only
supports the Proleptic Gregorian calendar.

You can read more about the `state`, including :py:func:`sympl.datetime` in
:ref:`Model State`.

Initialize Components
---------------------

Now we need the objects that will process the state to move it forward in time.
Those are the "components":

.. code-block:: python

    physics_stepper = AdamsBashforth([
        UpdateFrequencyWrapper(Radiation(), timedelta(hours=2)),
        BoundaryLayer(),
        DeepConvection(),
    ])
    implicit_dynamics = ImplicitDynamics()

:py:class:`~sympl.AdamsBashforth` is a :py:class:`~sympl.TimeStepper`, which is
created with a set of :py:class:`~sympl.Prognostic` components.
The :py:class:`~sympl.Prognostic` components we have here are `Radiation`,
`BoundaryLayer`, and `DeepConvection`. Each of these carries information about
what it takes as inputs and provides as outputs, and can be called with a model
state to return tendencies for a set of quantities. The
:py:class:`~sympl.TimeStepper` uses this information to step the model state
forward in time.

The :py:class:`~sympl.UpdateFrequencyWrapper` applied to the `Radiation` object
is an object that acts like a :py:class:`~sympl.Prognostic` but only computes
its output if at least a certain amount of model time has passed since the last
time the output was computed. Otherwise, it returns the last computed output.
This is commonly used in atmospheric models to avoid doing radiation
calculations (which are very expensive) every timestep, but it can be applied
to any Prognostic.

The :py:class:`ImplicitDynamics` class is a :py:class:`~sympl.Implicit` object, which
steps the model state forward in time in the same way that a :py:class:`~sympl.TimeStepper`
would, but doesn't use :py:class:`~sympl.Prognostic` objects in doing so.

The Main Loop
-------------

With everything initialized, we have the part of the code where the real
computation is done -- the main loop:

.. code-block:: python

    timestep = timedelta(minutes=30)
    while state['time'] < datetime(2010, 1, 1):
        physics_diagnostics, state_after_physics = physics_stepper(state, timestep)
        dynamics_diagnostics, next_state = implicit_dynamics(state_after_physics, timestep)
        state.update(physics_diagnostics)
        state.update(dynamics_diagnostics)
        plot_monitor.store(state)
        next_state['time'] = state['time'] + timestep
        state = next_state

In the main loop, a series of component calls update the state, and the figure
presented by ``plot_monitor`` is updated. The code is meant to be as
self-explanatory as possible. It is necessary to manually set the time of the
next state at the end of the loop. This is not done automatically by
:py:class:`~sympl.TimeStepper` and :py:class:`~sympl.Implicit` objects, because
in many models you may want to update the state with multiple such objects
in a sequence over the course of a single time step.
