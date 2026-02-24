=====================================
Sympl: A System for Modelling Planets
=====================================

**sympl** is an open source project aims to enable researchers and other
users to write understandable, modular, accessible Earth system and planetary
models in Python. It is meant to be used in combination with other packages
that provide model components in order to write model scripts. Its source
code can be found on GitHub_.

New users should read the :ref:`quickstart`. This framework is meant to be used along
with model toolkits like CliMT_ to write models. See the paper_ on Sympl and
CliMT for a good overview and some examples! You may also want to check out the
`CliMT documentation`_.

.. _GitHub: https://github.com/mcgibbon/sympl
.. _paper: https://www.geosci-model-dev.net/11/3781/2018/gmd-11-3781-2018.html
.. _`CliMT documentation`: https://climt.readthedocs.io/en/latest/
.. _CliMT: https://github.com/climt/climt

Documentation
-------------

.. toctree::
   :caption: Users
   :maxdepth: 1

   overview
   quickstart
   faq
   installation
   state
   constants
   timestepping
   computation
   backend
   monitors
   composites
   tracers
   units
   writing_components
   memory_management
   contributing
   history
   authors

..   installation

License
-------

**sympl** is available under the open source `BSD License`_.

.. _BSD License: https://github.com/mcgibbon/sympl/blob/master/LICENSE
