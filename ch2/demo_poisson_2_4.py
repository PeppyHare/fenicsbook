#!/usr/bin/env python
'''
This is a docstring for demo_poisson_2_4

'''

from mshr import *
from fenics import *
import numpy as np
import matplotlib.pyplot as plt


class PoissonDemo24():
    def __init__(self):
        # Define mesh and function space
        domain = Circle(Point(0, 0), 1)
        mesh = generate_mesh(domain, 64)
        V = FunctionSpace(mesh, 'P', 2)

        # Define boundary condition
        # Our pressure function is:
        # p(x, y) = A / ( 2 \pi \sigma ) exp(-1/2((x - x_0)/\sigma)^2) - 1/2((y - y_0)/\sigma)^2)
        #
        beta = 8
        R0 = 0.0
        p = Expression(
            '4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
            degree=1,
            beta=beta,
            R0=R0
        )

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, Constant(0), boundary)

        w = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(w), grad(v)) * dx
        L = p * v * dx

        w = Function(V)
        solve(a == L, w, bc)

        p = interpolate(p, V)
        plot(w, title='Deflection')
        # plot(p, title='Load')

        plt.show()
