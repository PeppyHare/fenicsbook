#!/usr/bin/env python3
r"""
Analysis of structures is one of the major activities of modern engineering, which likely makes the PDE modeling the deformation of elastic bodies the most popular PDE in the world. It takes just one page of code to solve the equations of 2D or 3D elasticity using FEniCS, as we show here.

**PDE Problem**

The equations governing small elastic deformations of a body :math:`\Omega` can be written as


.. math::
    - \nabla \cdot \sigma & = & f \text{ in } \Omega \\
    \sigma & = & \lambda \text{tr}(\epsilon) I + 2 \mu \epsilon \\
    \epsilon & = & \frac{1}{2} \left( \nabla u + (\nabla u)^\top \right) 

where :math:`\sigma` is the stress tensor, :math:`f` is the body force per unit volume, :math:`\lambda` and :math:`\mu` are Lamé's elasticity parameters for the material in :math:`\Omega`, :math:`I` is the identity tensor, tr is the trace operator on a tensor, :math:`\epsilon` is the symmetric strain-rate tensor (symmetric gradient), and :math:`u` is the displacement vector field. We have here assumed isotropic conditions.

We combine the above to obtain

.. math::
    \sigma = \lambda (\nabla \cdot u) I + \mu \left( \nabla u + (\nabla u)^\top \right)

We can easily obtain a single vector PDE for :math:`u`, which is the governing PDE for the unknown (Navier's equation). As it turns out, for the variational formulation it is convenient to keep the equations split as above

**Variational Formulation**

"""

from fenics import *


class ClampedBeam(object):
    r"""
    """

    def __init__(self):
        mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
        V = VectorFunctinoSpace(mesh, 'P', 1)


if __name__ == '__main__':
    ClampedBeam()
