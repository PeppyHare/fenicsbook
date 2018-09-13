#!/usr/bin/env python
r'''

2.4: Problem Description
+++++++++++++++++++++++++++++++

After kicking the tires with a test problem where we know the answer, we turn to a physically more relevant problem with solutions of a somewhat more exciting shape.

We want to compute the deflection :math:`D(x, y)` of a two-dimensional circular membrane of radius :math:`R`, subject to a load :math:`p` over the membrane. The appropriate PDE model is


.. math::
    -T \nabla ^2 D = p \quad \text{ in } \Omega = \{ (x, y) | x^2 + y^2 \leq R \}

Here :math:`T` is the tension in the membrane (constant), :math:`p` is the external pressure load. The boundary of the membrane has no deflection, implying :math:`D=0` as a boundary condition. We'll model a localized load as a Gaussian:


.. math::
    p(x, y) = \frac{A}{2\pi \sigma} \exp \left( - \frac{1}{2} \left( \frac{x - x_0}{\sigma} \right)^2  - \frac{1}{2} \left( \frac{y -y_0}{\sigma}  \right)^2 \right)

The parameter :math:`A` is the amplitude of the pressure, :math:`(x_0, y_0)` the localization of the maximum point of the load, and :math:`\sigma` the "width" of the load. We will take the center of the pressure to :math:`(0, R_0)` for some :math:`0 < R_0 < R`

Scaling the Equation
+++++++++++++++++++++

We have a lot of physics parameters in the problem, and as with any such problem we can improve our numerical precision by grouping them by means of scaling. We introduce dimensionless coordinates :math:`\bar{x} = x / R, \, \bar{y} = y / R` and a dimensionless deflection :math:`w = D / D_c` where :math:`D_C` is a characteristic size of the deflection. Introducing :math:`\bar{R_0} = R_0 / R` we obtain

.. math::
    - \frac{\partial ^2 w }{\partial \bar{x} ^2} - \frac{\partial ^2 w }{\partial \bar{y} ^2} = \alpha \exp \left( - \beta ^2 (\bar{x}^2 + (\bar{y} - R_0)^2 )  \right) \quad \text{ for } \bar{x}^2 + \bar{y}^2 < 1

where

.. math::
    \alpha = \frac{R^2A}{2 \pi T D_c \sigma}, \quad \beta = \frac{R}{\sqrt{2} \sigma}  

With the appropriate scaling, :math:`w` and its derivatives are of size unity, so the LHS of the scaled PDE is about unity in size, while the right hand side has :math:`\alpha` as its characteristic size. This suggests choosing :math:`\alpha` to be unity, or around unity. We shall in this particular case choose :math:`\alpha = 4` (One can also find the analytical solution in scaled coordinates and show that the maximum deflection :math:`D(0, 0)` is :math:`D_c` if we choose :math:`\alpha = 4` to determine :math:`D_c`). With :math:`D_c = A R^2  / (8 \pi \sigma T)` and dropping the bars for convenience we obtain the scaled problem


.. math::
    - \nabla ^2 w = 4 \exp \left( - \beta^2(x^2+(y-R_0)^2) \right)

to be solved over the unit disc with :math:`w = 0` on the boundary. Now there are only two parameters to vary: the dimensionless extent of the pressure :math:`\beta` and the localization of the pressure peak :math:`R_0 \in [0, 1]`. As :math:`\beta \rightarrow 0`, the solution will approach the special case :math:`w = 1 - x^2 - y^2`

Given a computed scaled solution :math:`w` the physical deflection can be computed by 

.. math::
    D = \frac{AR^2}{8 \pi \sigma T} w 


'''

from mshr import *
from fenics import *
import numpy as np
import matplotlib.pyplot as plt


class PoissonDemo24():
    r'''
    Solving this problem is very similar to the previous test problem, with just a few modifications.

    **Defining a unit disk mesh:**

    A mesh over the unit disk can be created by the :code:`mshr` tool in FEniCS. The :code:`Circle` shape from :code:`mshr` takes the center and radius of the circle as arguments. The second argument to :code:`generate_mesh` specifies the desired mesh resolution. The cell size will be (approximately) equal to the diameter of the domain divided by the resolution.

    **Defining the load:**

    We use an :code:`Expression` object to represent the pressure function in our PDE. We set the physical parameters :math:`\beta` and :math:`R_0` by keyword arguments. The coordinates in Expression objects are always an array :code:`x` with components :code:`x[0], x[1], x[2]` corresponding with :math:`x`, :math:`y`, and :math:`z`. Otherwise we are free to introduce names of parameters as long as these are given default values by keyword arguments. All the parameters initialized by keyword arguments can at any time have their values modified.

    .. code-block:: python

        p.beta = 12
        p.R0 = 0.3


    **Defining the variational problem:**

    The variational problem and boundary conditions are the same as in our first Poisson problem, but we introduce :code:`w` instead of :code:`u` as primary unknown and :code:`p` instead of :code:`f` as right-hand side function to better align with our problem description.


    .. code-block:: python

        w = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(w), grad(v)) * dx
        L = p * v * dx

        w = Function(V)
        solve(a == L, w, bc)


    '''

    def __init__(self, n, beta=8, R0=0.6):
        # Define mesh and function space
        domain = Circle(Point(0, 0), 1)
        mesh = generate_mesh(domain, n)
        V = FunctionSpace(mesh, 'P', 2)
        p = Expression(
            '4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
            degree=1,
            beta=beta,
            R0=R0
        )

        # Parameterized values can be updated as `p.beta = 12` or `p.R0 = 0.3`

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


if __name__ == '__main__':
    PoissonDemo24(n=64, beta=8, R0=0.6)
