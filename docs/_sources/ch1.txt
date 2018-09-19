Chapter 1: Intro
------------------

Running FEniCS
~~~~~~~~~~~~~~~~

I've found that using the "FEniCS in Docker" setup yields great results, and building FEniCS from source is extremely painful. Simply follow https://fenics.readthedocs.id/projects/containers/en/latest to get started

Building the docs
~~~~~~~~~~~~~~~~~~~

To build these docs, first start up a new fenics project docker instance, with something like::

    brew install fenicsproject
    fenicsproject create myspace
    fenicsproject start myspace

Once inside, the docs are served by running ``filewatcher.sh``, which will watch the project directory and re-build if there are any changes. The local port to watch will depend on which port the fenics container is exposed on.