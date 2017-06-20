==========================
Frequently Asked Questions
==========================

Isn't Python too slow for Earth System Models?
----------------------------------------------

Not in general. Most model run time is spent within code such as the dynamical
core, radiation parameterization, and other physics parameterizations. These
components can be written in your favorite compiled language like Fortran or
C, and then run from within Python. For new projects where you're writing a
component from scratch, we recommend **Cython**, as it allows you to write typed
Python code which gets converted into C code and then compiled. Sympl is
designed so that only overhead tasks need to be written in Python.

If 90% of a model's run time is spent within this computationally intensive,
compiled code, and the other 10% is spent in overhead code, then that overhead
code taking 3x as long to run would only increase the model's run time by 1/5th.

But the run time of a model isn't the only important aspect, you also have to
consider time spent programming a model. Poorly designed and documented code
can cost weeks of researcher time. It can also take a long time to perform
tasks that Sympl makes easy, like porting a component from one model to
another. Time is also saved when others have to read and understand your model
code.

In short, the more your work involves configuring and developing models, the
more time you will save, at the cost of slightly slower model runs. But in the
end, what is the cost of your sanity?

What calendar is my model using?
--------------------------------

Hopefully the section on :ref:`Choice of Datetime` can clear this up.
