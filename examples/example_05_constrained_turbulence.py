import matplotlib.pyplot as plt
import numpy as np

from mannrs import ConstrainedStencil, Constraint


params = {}
Lx = 1000
Nx = 300
N_periods = 3
N_constraints = 100


def random_walk(n_steps, std_dev, seed=None):
    if seed:
        np.random.seed(seed)
    # Generate random steps with standard deviation

    steps = np.random.normal(loc=0, scale=std_dev, size=n_steps)
    # Calculate the position by cumulative sum of steps
    position = np.cumsum(steps)

    return position - position.mean()


if __name__ == "__main__":
    x_constraint = np.linspace(0, Lx, N_constraints)
    y_constraint = 10 * np.sin(N_periods * np.pi * x_constraint / Lx) + 5 * np.sin(
        5.2 * N_periods * np.pi * x_constraint / Lx
    )

    constraints = [
        Constraint(x, 100, 100, y) for x, y in zip(x_constraint, y_constraint)
    ]

    stencil = ConstrainedStencil(
        constraints=constraints,
        L=30.0,
        gamma=3.2,
        Lx=Lx,
        Ly=200,
        Lz=200,
        Nx=Nx,
        Ny=32,
        Nz=32,
        aperiodic_x=True,
        aperiodic_y=True,
        aperiodic_z=True,
        parallel=True,
        corr_thres=0.001,
        sinc_thres=12.0,
    )

    N_boxes = 10
    fig, axes = plt.subplots(1, N_boxes)
    x = np.linspace(0, Lx, Nx)
    axes = np.atleast_1d(axes)
    ys = []
    for i in range(N_boxes):
        U, V, W = stencil.turbulence(
            ae=0.2, seed=i, impulse_thres=0.0001, parallel=True
        )
        ys.append(U[:, 16, 16])
        axes[i].imshow(U[:, :, 16])
    plt.savefig("constraint_slice.png", dpi=300, bbox_inches="tight")

    plt.figure()

    for y in ys:
        plt.plot(x, y)
    plt.plot(x_constraint, y_constraint, ".k")
    plt.savefig("constraint_timeseries.png", dpi=300, bbox_inches="tight")
