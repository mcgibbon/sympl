==============
Array Backends
==============

Historically, Sympl was tightly coupled with `xarray` and `numpy`. While this
combination provides a powerful and intuitive interface for handling metadata
and multidimensional arrays, it can be restrictive for users who want to
leverage modern computational frameworks like JAX or PyTorch, or those who
prefer different metadata containers.

To address this, Sympl now includes an array backend abstraction. This allows
Sympl to be used with different array types and metadata containers while
maintaining its core functionality of dimension and unit handling.

Motivations
-----------

The primary goals of the array backend abstraction are:

* **Framework Interoperability**: Enable Sympl to work seamlessly with various
  computational frameworks (e.g., JAX, PyTorch, Dask) by allowing them to
  provide their own array types to components.
* **Flexibility**: Allow users to choose different metadata containers or unit
  libraries if `xarray` and `Pint` do not meet their needs.
* **Decoupling**: Separate the core logic of Sympl (component coordination,
  dimension/unit matching) from the specific data structures used to store
  and manipulate arrays.

Current Implementation
----------------------

The backend system is built around an abstract base class :py:class:`~sympl.StateBackend`.

.. autoclass:: sympl.StateBackend
    :members:

Default Backend
***************

The default backend in Sympl is :py:class:`~sympl.DataArrayBackend`, which
continues to use `xarray.DataArray` as the primary state container and `numpy`
for raw arrays passed to components.

.. autoclass:: sympl.DataArrayBackend
    :members:

Using Backends
--------------

You can check the current backend or set a new one using
:py:func:`~sympl.get_backend` and :py:func:`~sympl.set_backend`.

.. code-block:: python

    import sympl
    from my_custom_backend import MyBackend

    # Get current backend
    current_backend = sympl.get_backend()

    # Set a new backend
    sympl.set_backend(MyBackend())

How it Works
------------

When a component is called (e.g., via its ``__call__`` method), Sympl uses the
active backend to prepare the model state for the component's ``array_call``.

1. **Extraction**: The backend's :py:meth:`~sympl.StateBackend.get_array` method
   is called for each required input. This method is responsible for
   converting units, aligning dimensions, and returning the raw array type
   expected by the component (e.g., a `numpy.ndarray` or a `jax.numpy.ndarray`).
2. **Execution**: The component's ``array_call`` is executed with these raw
   arrays.
3. **Wrapping**: After the component returns its results, the backend's
   :py:meth:`~sympl.StateBackend.create_quantity` method is used to wrap the
   returned raw arrays back into the backend's preferred container (e.g.,
   a :py:class:`~sympl.DataArray`).

This process ensures that component authors can focus on writing their
computational logic using standard array interfaces, while Sympl and the
backend handle the complexities of metadata and data transformation.

Implementing a Custom Backend
-----------------------------

To implement a custom backend, you must create a subclass of
:py:class:`~sympl.StateBackend` and implement all of its abstract methods.
For example, if you wanted to create a backend that uses JAX arrays
directly, you would implement :py:meth:`~sympl.StateBackend.get_array` to
return JAX arrays and :py:meth:`~sympl.StateBackend.create_quantity` to wrap
them in your preferred metadata container.
