import matplotlib.pyplot as plt
import numpy as np

from mannrs import ConstrainedStencil, Constraint

params = {}
Lx = 2048.0
Nx = 1024


if __name__ == "__main__":
    x_constraint = np.arange(0, 2000, 2)
    u_constraint = 10 * np.sin(x_constraint / 10)
    constraints = [Constraint(x, 60, 0, u) for x, u in zip(x_constraint, u_constraint)]
    print(len(constraints))
    stencil = ConstrainedStencil(
        constraints=constraints,
        ae=1.0,
        L=29.4,
        gamma=3.9,
        Lx=Lx,
        Ly=4 * 32,
        Lz=4 * 32,
        Nx=Nx,
        Ny=32,
        Nz=32,
        parallel=True,
    )
    print(stencil.stencil)

    U, V, W = stencil.turbulence(1234, parallel=True)
