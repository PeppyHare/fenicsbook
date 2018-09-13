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


Chapter 2: Fundamentals
--------------------------

Here we take a brief peek at the most basic FEniCS capabilities by solving the Poisson equation, the "hello world" of PDEs.


2.1 - 2.3 Finite Element Variational Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ch2.demo_poisson
    :members:
    :noindex:

2.4 Deflection of a Membrane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ch2.demo_poisson_2_4
    :members:
    :noindex:

Chapter 3: A Gallery of finite element solvers
------------------------------------------------

The goal of this chapter is to demonstrate how a range of important PDEs from science and engineering can be quickly solved with a few lines of FEniCS code. We start with the heat equation and continue with a nonlinear Poisson equation, the equations for linear elasticity, the Navier–Stokes equations, and finally look at how to solve systems of nonlinear advection–diffusion–reaction equations. These problems illustrate how to solve time-dependent problems, nonlinear problems, vector-valued problems, and systems of PDEs. For each problem, we derive the variational formulation and express the problem in Python in a way that closely resembles the mathematics.

3.1 The Heat Equation
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ch3.heat_equation
    :members:
    :noindex:
