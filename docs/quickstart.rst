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
    from sympl import AdamsBashforth, PlotFunctionMonitor
    from datetime import datetime, timedelta

    set_prognostic_update_frequency(Radiation, timedelta(hours=2))

    def my_plot_function(fig, state):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_title('Lowest model level air temperature (K)')
        im = ax.pcolormesh(
            state['air_temperature'].to_units('degK').values[0, :, :],
            vmin=260.
            vmax=310.)
        cbar = fig.colorbar(im)

    plot_monitor = PlotFunctionMonitor(my_plot_function)

    state = get_initial_state(nx=256, ny=128, nz=64)
    state['time'] = datetime(2000, 1, 1)

    physics_stepper = AdamsBashforth(
        Radiation(),
        BoundaryLayer(),
        DeepConvection(),
    )
    implicit_dynamics = ImplicitDynamics()

    timestep = timedelta(minutes=30)
    while state['time'] < datetime(2010, 1, 1):
        physics_diagnostics, state_after_physics = physics_stepper(state)
        dynamics_diagnostics, next_state = implicit_dynamics(state_after_physics)
        state_after_physics.update(physics_diagnostics)
        state_after_physics.update(dynamics_diagnostics)
        plot_monitor.store(state_after_physics)
        next_state['time'] = state['time'] + timestep
        state = next_state

Importing Packages
------------------

At the beginning of the script we have import statements:

.. code-block:: python

    from model_package import (
        get_initial_state, Radiation, BoundaryLayer, DeepConvection,
        ImplicitDynamics)
    from sympl import AdamsBashforth, PlotFunctionMonitor
    from datetime import datetime, timedelta

These grant access to the objects that will be used to construct the model,
and are dependent on the model package you are using. Here, the names
`model_package`, `get_initial_state`, `Radiation`, `BoundaryLayer`,
`DeepConvection`, and `ImplicitDynamics` are placeholders, and do not refer to
an actual existing package.

Modifying Classes
-----------------

The following line modifies the Radiation class so that it will only compute
tendencies every 2 hours.

.. code-block::python

    set_prognostic_update_frequency(Radiation, timedelta(hours=2))

Radiation is a :py:class:`sympl.Prognostic` class, which means it takes in a
model state and returns tendencies from that state. The
:py:function:`sympl.set_prognostic_update_frequency` function modifies the
class so that when it is given a state, it checks whether its given amount of
(model) time has passed since the last time it computed tendencies, and if
not it returns the cached tendencies which it last computed.

Defining a PlotFunctionMonitor
------------------------------

Here we define a plotting function, and use it to create a
:py:class:`sympl.Monitor` using :py:class:`sympl.PlotFunctionMonitor`:

.. code-block::python

    def my_plot_function(fig, state):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_title('Lowest model level air temperature (K)')
        im = ax.pcolormesh(
            state['air_temperature'].to_units('degK').values[0, :, :],
            vmin=260.
            vmax=310.)
        cbar = fig.colorbar(im)

    plot_monitor = PlotFunctionMonitor(my_plot_function)

That `Monitor` will be used to produce an animated plot of the lowest model
level air temperature as the model runs. Here we assume that the first axis
is the vertical axis, and that the lowest level is at the lowest index, but
this might be different for different models.

Initialize the Model State
--------------------------

To initialize the model, we need to create a dictionary which contains the
model state. The way this is done is model-dependent. Here we assume there is
a function that was defined by the `model_package` package which does so:

.. code-block::python

    state = get_initial_state(nx=256, ny=128, nz=64)
    state['time'] = datetime(2000, 1, 1)

An initialized `state` is a dictionary whose keys are strings (like
'air_temperature') and values are :py:class:`sympl.DataArray` objects, which
store not only the data but also metadata like units. The one exception
is the "time" quantity which is either a `datetime`-like or `timedelta`-like
object. You can read more about the `state` in :ref:`Model State`.

Initialize Components
---------------------

Now we need the objects that will process the state to move it forward in time.
Those are the "components":

.. code-block::python

    physics_stepper = AdamsBashforth(
        Radiation(),
        BoundaryLayer(),
        DeepConvection(),
    )
    implicit_dynamics = ImplicitDynamics()

:py:class:`sympl.AdamsBashforth` is a :py:class:`sympl.TimeStepper`, which is
created with a set of :py:class:`sympl.Prognostic` components. The `Prognostic`
components we have here are `Radiation`, `BoundaryLayer`, and
`DeepConvection`. Each of these carries information about what it takes
as inputs and provides as outputs, and can be called with a model state
to return tendencies for a set of quantities. The `TimeStepper` uses
this information to step the model state forward in time.

The `ImplicitDynamics` class is a :py:class:`sympl.Implicit` object, which
steps the model state forward in time in the same way that a `TimeStepper`
would, but doesn't use `Prognostic` objects in doing so.

The Main Loop
-------------

With everything initialized, we have the part of the code where the real
computation is done -- the main loop:

.. code-block::python

    timestep = timedelta(minutes=30)
    while state['time'] < datetime(2010, 1, 1):
        physics_diagnostics, state_after_physics = physics_stepper(state)
        dynamics_diagnostics, next_state = implicit_dynamics(state_after_physics)
        state_after_physics.update(physics_diagnostics)
        state_after_physics.update(dynamics_diagnostics)
        plot_monitor.store(state_after_physics)
        next_state['time'] = state['time'] + timestep
        state = next_state
