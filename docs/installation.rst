.. highlight:: shell

============
Installation
============

Latest release
--------------

To install Sympl, run this command in your terminal:

.. code-block:: console

    $ pip install sympl

This is the preferred method to install Sympl, as it will always install the
most recent release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Sympl can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/mcgibbon/sympl

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/mcgibbon/sympl/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

If you are looking to modify the code, you can install it with:

.. code-block:: console

    $ python setup.py develop

This configures the package so that Python points to the current directory
instead of copying files. Then when you make modifications to the source code
in that directory, they are automatically used by any new Python sessions.

.. _Github repo: https://github.com/mcgibbon/sympl
.. _tarball: https://github.com/mcgibbon/sympl/tarball/master
