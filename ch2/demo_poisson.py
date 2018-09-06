#!/usr/bin/env python
r'''
Introduction to the finite element variational method!

The "hello world" of the finite element methods for PDEs is the Poisson equation, which consists of the following boundary value problem:

.. math::
    - \nabla ^2 u(x) = f(x) : \quad x \text{  in  } \Omega \\
    u(x) = u_D(x) : \quad x \text{  on  } \partial \Omega

Here u is our unknown function, f = f(x) is a prescribed function, :math:`\nabla^2` is the Laplace operator, :math:`\Omega` is the spatial domain, and :math:`\partial \Omega` is the boundary of :math:`\Omega`


Solving such a boundary-value problem in fenics involves:

1. Identify the computational domain, the PDE, its boundary conditions, and source terms (f).

2. Reformulate the PDE as a finite element variational problem.

3. Write a Python program which defines the computational domain, the variational problem, the boundary conditions, and source terms using the corresponding FEniCS abstractions.

4. Call FEniCS to solve the boundary-value problem and, optionally, extend the program to compute derived quantities such as fluxes and averages, and visualize the results.


Variational Formulation
++++++++++++++++++++++++++

We'll need a brief introduction to the variational method here. The basic recipe for turning a PDE into a variational problem is to multiply the PDE by a vunction v, integrate the resulting equation over the domain :math:`\Omega`, and perform integration by parts of terms with second-order derivatives. The function v which multiplies the PDE is called a *test function*. The unknown function *u* to be approximated is referred to as a *trial function*. The terms trial and test functions are used in FEniCS programs too. The trial and test functions belong to certain function spaces that specify the properties of the functions.
'''

from fenics import *
import numpy as np
import matplotlib.pyplot as plt


class PoissonDemo():
    r'''
    This is a docstring for demo_poisson

    Here we are trying to solve a problem we already know the answer to. The solution we already know to be:

    .. math::
        u_D = 1 + x^2 + 2y^2
    '''

    def __init__(self, n):
        # Create mesh and define function space
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, 'P', 2)

        # Define boundary condition
        u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

        tol = DOLFIN_EPS

        def boundary(x):
            return near(x[0], 0, tol) or near(x[1], 0, tol) or \
                near(x[0], 1, tol) or near(x[1], 1, tol)

        bc = DirichletBC(V, u_D, boundary)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(-6.0)
        a = dot(grad(u), grad(v)) * dx
        L = f * v * dx

        # Compute solution
        u = Function(V)
        solve(a == L, u, bc)

        # Plot solution and mesh
        plot(u)
        plot(mesh)

        # Save solution to file in VTK format
        vtkfile = File('poisson/solution.pvd')
        vtkfile << u

        # Compute error in L2 norm
        error_L2 = errornorm(u_D, u, 'L2')

        # Compute maximum error at vertices
        vertex_values_u_D = u_D.compute_vertex_values(mesh)
        vertex_values_u = u.compute_vertex_values(mesh)
        error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

        # Print errors
        print('Error_L2 = ', error_L2)
        print('error_max = ', error_max)

        plt.show()


if __name__ == '__main__':
    PoissonDemo(8)
