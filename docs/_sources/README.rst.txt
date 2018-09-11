.. 2016book documentation master file, created by
   sphinx-quickstart on Mon Sep  3 23:41:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to 2016book's documentation!
====================================

This is where I'll keep notes as I try to work through the `2016 version of the FEniCS tutorial <https://www.springer.com/gp/book/9783319524610>`_. The FEniCS project is a lovely open-source project which aims to provide powerful PDE solvers that don't necessarily require impossible-to-read syntax in the source code!

To build these docs, first start up a new fenics project docker instance, with something like::

    brew install fenicsproject
    fenicsproject create myspace
    fenicsproject start myspace

Once inside, the docs are served by running ``filewatcher.sh``, which will watch the project directory and re-build if there are any changes. The local port to watch will depend on which port the fenics container is exposed on.


Chapter 2
------------------


.. automodule:: ch2
    :members:
    :noindex:


2.1 Finite Element Variational Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ch2.demo_poisson
    :members:
    :noindex:

Poisson2_4
~~~~~~~~~~~~~~

.. automodule:: ch2.demo_poisson_2_4
    :members:
    :noindex:

Chapter 3
------------------

.. automodule:: ch3
    :members:
    :noindex:

Chapter 3
~~~~~~~~~~~~~~

.. automodule:: ch3.heat_equation
    :members:
    :noindex:
