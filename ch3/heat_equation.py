#!/usr/bin/env python3
r'''
As our first extension of the Poisson problem, we consider the time-dependent heat equation, or the time-dependent diffusion equation. 

The PDE problem:
++++++++++++++++++

Our model problem for time-dependent PDEs reads

.. math::

    \frac{\partial u}{\partial t} = \nabla^2u + f & \quad & \text{ in } \Omega \times (0, T] \\
    u = u_D & \quad & \text{ on } \partial \Omega \times (0, T] \\
    u = u_0 & \quad & \text{ at } t = 0

Here, :math:`u` varies with space *and* time. The source functiona nd the boundary values may also vary with space and time. The initial condition :math:`u_0` is a function of space only.

Variational Formulation
++++++++++++++++++++++++
A straightforward approach to solving time-dependent PDEs by the finite element method is to first discretize the time derivative by a finite difference approximation, which yields a sequence of stationary problems, and then in turn each stationary problem into a variational formulation.

Let superscript :math:`n` denote a quantity at time :math:`t_n` where :math:`n` is an integer counting time levels. For example, :math:`u^n` means :math:`u` at time level :math:`n`. A finite difference discretization in time first consists of samling the PDE at some level, say :math:`t_{n+1}`:


.. math::
    \left( \frac{\partial u}{\partial t}  \right)^{n + 1} = \nabla ^2 u^{n + 1} + f ^{n+1}

The time-derivative can be approximated by a difference quotient. For simplicity and stability reasons, we choose a simple backward difference:


.. math::
    \left( \frac{\partial u}{\partial t} \right)^{n+1} \approx \frac{u^{n+1} - u^n}{\Delta t} 

where :math:`\Delta t` is the time discretization parameter. Combining these two expressions we get


.. math::
    \frac{u^{n+1} - u^n}{\Delta t}  = \nabla ^2 u^{n+1} + f^{n+1}

This is our time-discrete version of the heat equation, a so-called backward Euler or "implicit Euler" discretization.

We may reorder so that the LHS contains the terms with the unknown :math:`u^{n+1}` and the RHS contains computed terms only. The result is a sequence of spatial (stationary) problems for :math:`u^{n+1}`, assuming :math:`u^{n}` is known from the previous time step:


.. math::
    u^{0}  & = & u_0 \\
    u^{n+1} - \Delta t \nabla ^2 u^{n+1} & = & u^n + \Delta t f^{n+1}, \quad n = 0, 1, 2

Given :math:`u_0` we can solve for :math:`u^0`, :math:`u^1`, :math:`u^2`, and so on.

An alternative which can be convenient in implementations is to collect all terms on one side of the equality sign:


.. math::
    u^{n+1} - \Delta t \nabla ^2 u^{n+1} -  u^n - \Delta t f^{n+1}= 0, \quad n = 0, 1, 2

We use a finite element method to solve :math:`u^{0} = u_0` and either of the above expressions. This requires turning the equations into weak forms. As usual we multiply by a test function :math:`v \in \hat{V}` and integrate second-derivatives by parts. Introducing the symbol :math:`u` for :math:`u^{n+1}` (which is natural in code), the resulting weak form can be conveniently written in the standard notation:


.. math::
    a(u, v) = L_{n+1}(v),

where


.. math::
    a(u, v) = \int_{\Omega} (uv + \Delta t \nabla u \cdot \nabla v) dx, \\
    L_{n+1} (v) = \int_{\Omega} (u^n + \Delta t f^{n+1}) v\, dx

In addition to the variational problem to be solved in each time step, we also need to approximate the initial condition. This equation can also be turned into a variational problem:


.. math::
    a_0(u, v) = L_0(v),

with


.. math::
    a_0(u, v) & = & \int_{\Omega} uv \, dx \\
    L_0(v) & = & \int_\Omega u_0 v \, dx

When solving this variational problem, :math:`u^0` becomes the :math:`L^2` projection of the initial value :math:`u_0` into the finite element space. The alternative is to construct :math:`u^0` by just interpolating the initial value :math:`u_0`; that is, if :math:`u^0 = \sum ^N _{j = 1} U{_j}{^0}\phi_j` we simply set :math:`U_j = u_0(x_j, y_j)` where :math:`(x_j, y_j)` are the coordinates of node number :math:`j`.
We refer to these two strategies as computing the initial condition by either "projection" or "interpolation". Both operations are easy to compute in FEniCS through a single statement, using either :code:`project` or :code:`interpolate` function. The most common choice is :code:`project` which computes an approximation to :math:`u_0`, but in some applications where we want to verify the code by reproducing exact solutions, one must use :code:`interpolate` (and we use such a test problem here!)

In summary, we thus need to solve the following sequence of variational problems to compute the finite element solution to the heat equation: find :math:`u^0 \in V` such that :math:`a_0(u^0, v) = L_0(v)` holds for all :math:`v \in \hat{V}`, and then find :math:`u^{n+1} \in V` such that :math:`a(u^{n+1}, v) = L_{n+1} (v)` for all :math:`v \in \hat{V}`, or alternatively, :math:`F_{n+1}(u^{n+1}, v) = 0` for all :math:`v \in \hat{V}`, for :math:`n = 0, 1, 2, \ldots`

'''

from fenics import *
import numpy as np


class ATestProblem(object):
    r"""
    Just as for the Poisson problem from the previous chapter, we construct a test problem that makes it easy to determine if the calculations are correct. Since we know that our first-order time-stepping scheme is exact for linear functions, we create a test problem which has linear variation in time. We combine this with a quadratic variation in space:


    .. math::
        u = 1 + x^2 + \alpha y^2 + \beta t

    which yields a function whose computed values at the nodes will be exact, regardless of the size of the elements and :math:`\Delta t`, as long as the mesh is uniformly partitioned. By inserting ^ into the heat equation, we find that the RHS :math:`f` must be given by :math:`f(x, y, t) = \beta - 2 - 2 \alpha`. The boundary value is :math:`u_D(x, y, t) = 1 + x^2 + \alpha y^2 + \beta t` and the initial value is :math:`u_0(x, y) = 1 + x^2 + \alpha y^2`.

    A new issue is how to deal with functions that vary in both space and time, such as our boundary condition here :math:`u_D(x, y, t) = 1 + x^2 + \alpha y^2 + \beta t`. A natural solutino is to use a FEniCS :code:`Expression` with time t as a parameter, in additional to the physical parameters:

    .. code-block:: python

        alpha = 3; beta = 1.2
        u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                         degree=2, alpha=alpha, beta=beta, t=0)

    We use the variable :code:`u` for the unknown :math:`u^{n+1}` at the new time step and the variable :code:`u_n` for :math:`u^n` at the previous time step. The initial value of :code:`u_n` can be computed by either projection or interpolation of :math:`u_0`. Since we set :code:`t = 0` for the boundary value :code:`u_D`, we can use :code:`u_D` to specify the initial condition:


    .. code-block:: python

        u_n = project(u_D, V)
        # or 
        u_n = interpolate(u_D, V)

    We can either define :math:`a` or :math:`L` according to the formulas we have above, or we may just define :math:`F` and ask FEniCS to figure out which terms should go into the bilinear form :math:`a` and which should go into the linear form :math:`L`. The latter is convenient, especially in more complicated problems, so we illustrate that construction of :math:`a` and :math:`L`:


    .. code-block:: python

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)

        F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
        a, L = lhs(F), rhs(F)

    Finally, we can perform the time-stepping in a loop


    .. code-block:: python

        fid = File("test_problem/solution.pvd")
        u = Function(V)
        t = 0
        for n in range(num_steps):
            # Update current time
            t += dt
            u_D.t = t
            # Compute solution
            solve(a == L, u, bc)
            fid << u, t
            u_n.assign(u)


    """

    def __init__(self, n_s=300, T=2.0, alpha=3, beta=1.2, steps=40):
        dt = T / steps

        # Create mesh and define function space
        nx = ny = n_s
        mesh = UnitSquareMesh(nx, ny)
        V = FunctionSpace(mesh, 'P', 1)

        # Define boundary condition
        u_D = Expression(
            '1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
            degree=2,
            alpha=alpha,
            beta=beta,
            t=0
        )

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u_D, boundary)

        # Define initial value
        u_n = interpolate(u_D, V)
        # u_n = project(u_D, V)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(beta - 2 - 2 * alpha)

        F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
        a, L = lhs(F), rhs(F)

        # Output plots to a file for animation
        fid = File("test_problem/solution.pvd")

        # Time-stepping
        u = Function(V)
        t = 0
        for n in range(steps):

            # Update current time
            t += dt
            u_D.t = t

            # Compute solution
            solve(a == L, u, bc)

            # See https://fenicsproject.org/qa/7334/bug-in-how-vtk-files-are-written-in-fenics-1-5-0/?show=7334#q7334
            u.rename("u", "u")
            fid << u, t

            # Update previous solution
            u_n.assign(u)


class BHeatEquation(object):
    r"""
    Thermal diffusion of a Gaussian fuction. We'd like to solve for the thermal diffusion of the following gaussian initial temperature distribution:

    .. math::
        u_0(x, y) = e^{-ax^2 -ay^2}

    with :math:`a = 5` on the domain :math:`[-2, 2] \times [2, 2]`. For this problem we will use homogeneous Dirichlet boundary conditions (:math:`u_D = 0`).

    The major changes required from our previous problem are: we now have a rectangular domain that isn't the unit square, so we use :code:`RectangleMesh`:


    .. code-block:: python

        nx = ny = 30
        mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)

    Note that we have used a much higher resolution than before to better resolve the features of the solution. We also need to define the initial condition and boundary condition. Both are easily changed by adding a new :code:`Expression` and by setting :math:`u = 0` on the boundary.
    """

    def __init__(self, n_s=300, T=2.0, a=5, steps=40):
        dt = T / steps

        # Create mesh and define function space
        nx = ny = n_s
        mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
        V = FunctionSpace(mesh, 'P', 1)

        # Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, Constant(0), boundary)

        # Define initial value
        u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=a)
        u_n = interpolate(u_0, V)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)

        F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
        a, L = lhs(F), rhs(F)

        # Create VTK file for saving solution
        fid = File('heat_gaussian/solution.pvd')

        # Time-stepping
        u = Function(V)
        t = 0
        for n in range(steps):
            # Update current time
            t += dt

            # Compute solution
            solve(a == L, u, bc)

            # See https://fenicsproject.org/qa/7334/bug-in-how-vtk-files-are-written-in-fenics-1-5-0/?show=7334#q7334
            # u.rename("u", "u")
            fid << u, t

            # Update previous solution
            u_n.assign(u)


if __name__ == "__main__":
    ATestProblem(n_s=8, T=2.0, alpha=5, beta=1.2, steps=10)
    # BHeatEquation(n_s=30, T=2.0, a=5, steps=50)
