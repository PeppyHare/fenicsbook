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

We'll need a brief introduction to the variational method here. The basic recipe for turning a PDE into a variational problem is to multiply the PDE by a vunction v, integrate the resulting equation over the domain :math:`\Omega`, and perform integration by parts of terms with second-order derivatives. The function v which multiplies the PDE is called a *test function*.The unknown function *u* to be approximated is referred to as a *trial function*. The terms trial and test functions are used in FEniCS programs too. The trial and test functions belong to certain function spaces that specify the properties of the functions.

For an example, we do just that for the Poisson equation

.. math::
    -\int _{\Omega} ( \nabla ^2 u)v \, dx = \int _{\Omega} f v \, dx

What we'd like to do is decrease the order of the derivatives of *u* and *v* as much as possible, so of course we'll be integrating by parts. To make the variational formulation work, we choose a function space such that the test function is required to vanish on the parts of the boundary where the solution *u* is known. This means that we get to drop the boundary terms, and we can pull off derivatives from *u* at the cost of a minus sign:

.. math::
    \int_{\Omega} \nabla u \cdot \nabla v \, dx = \int_{\Omega} f v \, dx

We can then define our original PDE as the variational problem: find :math:`v \in V` such that

.. math::
    \int_{\Omega} \nabla u \cdot \nabla v \, dx = \int_{\Omega} f v \, dx \quad \forall v \in \hat{V}

where the trial and test spaces :math:`V` and :math:`\hat{V}` are in the present problem defined as

.. math::
    V & = & { v \in H^1(\Omega) : v = u_D \text{ on } \partial \Omega } \\
    \hat{V} & = & { v \in H^1(\Omega) : v = 0 \text{ on } \partial \Omega }

Our finite element solver finds an approximate solution to this problem by replacing the infinite-dimentional function spaces by discrete trial and test spaces. Once we're there, voila! FEniCS can take care of the rest.

Abstract variational formulation
+++++++++++++++++++++++++++++++++++

It's convenient to introduce some notation for variational problems: find :math:`u \in V` such that

.. math::
    a(u, v) = L(v) \quad \forall v \in \hat{V}

In our example of the Poisson equation, we have:

.. math::
    a(u, v) & = & \int_{\Omega} \nabla u \cdot \nabla v \, dx \\
    L(v) & = & \int_\Omega f v \, dx

Here we say :math:`a(u, v)` is a *bilinear form* and :math:`L(v)` is a *linear form*. In each problem we want to solve, we'll identify the terms with the unknown *u* and collect them in :math:`a(u, v)`, and similarly collect all terms with only known functions in :math:`L(v)`.
'''

from fenics import *
import numpy as np
import matplotlib.pyplot as plt


class PoissonDemo():
    r'''

    Here we are trying to solve a problem we already know the answer to. Solutions that are low-order polynomials are great candidates to check the accuracy of our solution, as standard finite element function spaces of degree *r* will exactly reproduce polynomials of degree *r*. We manufacture some quadratic function in 2D as our exact solution, say

    .. math::
        u_e(x, y) = 1 + x^2 + 2y^2

    By inserting this into the Poisson equation we find that it is a solution if 

    .. math::
        f(x, y) & = & -6 \\
        u_D(x, y) & = & u_e(x, y) = 1 + x^2 + 2y^2

    For simplicity, we'll deal with the unit square as our domain

    .. math::
        \Omega = [0,1] \times [0,1]

    The code in this module shows how to solve this example problem in FEniCS, and since we already know the answer, we also compute the L2 error of our solution. Since we expect our discrete space to exactly reproduce the solution, the error should be within machine precision.
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
