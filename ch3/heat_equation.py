#!/usr/bin/env python3
"""
This is a module docstring
"""

from fenics import *
import numpy as np


class HeatEquation(object):
    """
    This is a class docstring in the heat_equation module

    Thermal diffusion of a Gaussian fuction. We'd like to solve for the diffusion of the following gaussian distribution

    .. math::
        u_0(x, y) = e^{-ax^2 -ay^2}
    """
    def __init__(self, num_steps, t, a):
        T = 2.0
        num_steps = 40
        dt = T / num_steps
        a = 5

        # Problem: thermal diffusion of a Gaussian hill
        #
        # u_0 = Exp(-a*x^2 - a*y^2)
        #
        #

        # Create mesh and define function space
        nx = ny = 300
        mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
        V = FunctionSpace(mesh, 'P', 1)

        bc = DirichletBC(V, Constant(0), boundary)

        # Define initial value
        u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5)
        u_n = interpolate(u_0, V)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)

        F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
        a, L = lhs(F), rhs(F)

        # Output plots to a file for animation
        fid = File("heat_equation/solution.pvd")

        # Time-stepping
        u = Function(V)
        t = 0
        for n in range(num_steps):

            # Update current time
            t += dt

            # Compute solution
            solve(a == L, u, bc)

            # See https://fenicsproject.org/qa/7334/bug-in-how-vtk-files-are-written-in-fenics-1-5-0/?show=7334#q7334
            # u.rename("u", "u")
            fid << u, t

            # Update previous solution
            u_n.assign(u)

    def boundary(x, on_boundary):
        # Define boundary condition
        return on_boundary


