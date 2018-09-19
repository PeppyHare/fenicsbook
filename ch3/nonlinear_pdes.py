#!/usr/bin/env python3
r"""

We now try to address how to solve nonlinear PDEs. By defining a nonlinear variational problem and calling the :code:`solve` function, they become just as easy as the linear ones. When we do so, we encounter a subtle difference in how the variational problem is defined.

**Model Problem**

As a model problem for the solution of nonlinear PDEs, we take the following nonlinear Poisson equation:


.. math::
    - \nabla \cdot (q(u) \nabla u ) = f

in :math:`\Omega` with :math:`u = u_D` on the boundary :math:`\partial \Omega`. The coefficient :math:`q = q(u)` makes the equation nonlinear (unless :math:`q(u)` is constant in :math:`u`).

**Variational Formulation**

As usual, we multiply our PDE by a test function :math:`v \in \hat{V}`, integrate over the domain, and integrate the second-order derivatives by parts. The boundary integral arising from integration by parts vanishes wherever we employ Dirichlet conditions. The resulting variational formulation of our model problem becomes: find :math:`u \in V` such that 


.. math::
    F(u;v) = 0 \quad \forall v \in \hat{V}

where


.. math::
    F(u;v) = \int_\Omega (q(u) \nabla u \cdot \nabla v - f v)\, dx

and

.. math::
    V & = & \{ v \in H^1(\Omega) : v = u_D \text{ on } \partial \Omega \} \\
    \hat{V} & = & \{ v \in H^1(\Omega) : v = 0 \text{ on } \partial \Omega \}

The discrete problem arises as usual by restricting :math:`V` and :math:`\hat{V}` to a pair of discrete spaces. As before, we omit any subscript on the discrete spaces and discrete solution. The discrete nonlinear problem is written as: find :math:`u \in V` such that

.. math::
    F(u;v) = 0 \quad \forall \, v \in \hat{V}

with :math:`u = \sum_{j=1}^N U_j \phi_j`. Since :math:`F` is nonlinear in :math:`u`, the variational statement gives rise to a system of nonlinear algebraic equations in the unknowns :math:`U_1, \ldots ,U_N`.

"""

# Warning: from fenics import * will import both ‘sym‘ and
# ‘q‘ from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *
# Use SymPy to compute f from the manufactured solution u
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


class ANonlinearTest(object):
    r"""
    For a nonlinear test problem, we need to choose the right-hand side :math:`f`, the coefficient :math:`q(u)` and the boundary value :math:`u_D`. Previously we have worked with manufactured solutions that can be reproduced without approximation errors. This is more difficult in nonlinear problems, and the algebra is more tedious.

    However we may utilize SymPy for symbolic computing and integrate such computations in the FEniCS solver. This allows us to easily experiment with different manufactured solutions. The forthcoming code with SymPy requires some basic familiarity with this package. In particular, we will use the SymPy functions :code:`diff` for symbolic differentiation and :code:`ccode` for C/C++ code generation.

    Our test problem here is :math:`q(u) = 1 + u^2`, and we define a two-dimensional manufactured solution that is linear in :math:`x` and :math:`y`.


    .. code-block:: python

        def q(u):
            "Return nonlinear coefficient"
            return 1 + u**2

        x, y = sym.symbols('x[0], x[1]')
        u = 1 + x + 2 * y
        f = -sym.diff(q(u) * sym.diff(u, x), x) - sym.diff(q(u) * sym.diff(u, y), y)
        f = sym.simplify(f)
        u_code = sym.printing.ccode(u)
        f_code = sym.printing.ccode(f)

    In SymPy we might normally write :code:`x, y = sym.symbols('x, y')`, but we want the resulting expressions to have valid syntax for FEniCS expression objects, so we use :code:`x[0]` and :code:`x[1]`.

    .. figure:: ch3/nonlinear_pdes_1.png

        Solution to the non-linear :math:`q(u) = 1 + u^2` with 8x8 spatial grid.

    """

    def __init__(self, n_s=8):
        def q(u):
            "Return nonlinear coefficient"
            return 1 + u**2

        x, y = sym.symbols('x[0], x[1]')
        u = 1 + x + 2 * y
        f = -sym.diff(q(u) * sym.diff(u, x), x) - sym.diff(q(u) * sym.diff(u, y), y)
        f = sym.simplify(f)
        u_code = sym.printing.ccode(u)
        f_code = sym.printing.ccode(f)
        print('u =', u_code)
        print('f =', f_code)

        mesh = UnitSquareMesh(n_s, n_s)
        V = FunctionSpace(mesh, 'P', 1)
        u_D = Expression(u_code, degree=1)

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u_D, boundary)

        u = Function(V)
        v = TestFunction(V)
        f = Expression(f_code, degree=1)
        F = q(u)*dot(grad(u), grad(v))*dx - f*v*dx
        solve(F == 0, u, bc)

        plot(u)
        plt.savefig('nonlinear_pdes_1.png')
        plt.show()


if __name__ == '__main__':
    ANonlinearTest(n_s=8)
