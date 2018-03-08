=========
Constants
=========

Configuration is an important part of any modelling framework. In Sympl,
component-specific configuration is given to components directly. However,
configuration values that may be shared by more than one component are stored
as constants. Good examples of these are physical constants, such as
gravitational_acceleration, or constants specifying processor counts.

Getting and Setting Constants
-----------------------------

You can retrieve and set constants using :py:func:`~sympl.get_constant` and
:py:func:`~sympl.set_constant`. :py:func:`~sympl.set_constant` will
allow you to set constants regardless of whether a value is already defined
for that constant, allowing you to add new constants we haven't thought of.

The constant library can be reverted to its original state when Sympl is
imported by calling :py:func:`~sympl.reset_constants`.

.. autofunction:: sympl.get_constant

.. autofunction:: sympl.set_constant

.. autofunction:: sympl.reset_constants

Condensible Quantities
----------------------

For Earth system modeling, water is used as a condensible compound. By
default, condensible quantities such as 'density_of_ice' and
'heat_capacity_of_liquid_phase' are aliases for the corresponding value for
water. If you would like to use a different condensible compound, you can
use the :py:func:`~sympl.set_condensible_name` function. For example:

.. code-block:: python

    import sympl
    sympl.set_condensible_name('carbon_dioxide')
    sympl.get_constant('heat_capacity_of_solid_phase', 'J kg^-1 K^-1')

will set the condensible compound to carbon dioxide, and then get the heat
capacity of solid carbon dioxide (if it has been set). For example, the constant
name 'heat_capacity_of_solid_phase' would then be an alias for
'heat_capacity_of_solid_carbon_dioxide'.

When setting the value of an alias, the value of the aliased quantity is the
one which will be altered. For example, if you run

.. code-block:: python

    import sympl
    sympl.set_constant('heat_capacity_of_liquid_phase', 1.0, 'J kg^-1 K^-1')

you would change the heat capacity of liquid water (since water is the default
condensible compound).

.. autofunction:: sympl.set_condensible_name

Default Constants
-----------------

The following constants are available in Sympl by default:

.. autoclass:: sympl._core.constants.ConstantList
