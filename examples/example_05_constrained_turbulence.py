import matplotlib.pyplot as plt
import numpy as np

from mannrs import ConstrainedStencil, Constraint

params = {}
Lx = 6000
Nx = 300


if __name__ == "__main__":
    x_constraint = np.linspace(0, Lx, 100)
    y_constraint = 10 * np.sin(x_constraint / 100) 
    constraints = [Constraint(x, 100, 0, y) for x, y in zip(x_constraint, y_constraint)]

    stencil = ConstrainedStencil(
        constraints=constraints,
        ae=2,
        L=30.0,
        gamma=3.2,
        Lx=Lx,
        Ly=200,
        Lz=200,
        Nx=Nx,
        Ny=32,
        Nz=32,
        parallel=True,
    )
    print(stencil.stencil)
    N_boxes = 1
    fig, axes = plt.subplots(1, N_boxes)
    x = np.linspace(0, Lx, Nx)
    axes = np.atleast_1d(axes)
    ys = []
    for i in range(N_boxes):
        U, V, W = stencil.turbulence(i, parallel=True)
        ys.append(U[:, 15, 0])
        axes[i].imshow(U[:, :, 0])

    plt.figure()

    for y in ys:
        plt.plot(x, y)
    plt.plot(x_constraint, y_constraint, ".k")
    plt.show()
