import matplotlib.pyplot as plt
import numpy as np
from rich import print

from mannrs.Constrained2 import ConstrainedStencil, Constraint

params = {}
Lx = 100000
Nx = 300
N_periods = 3
N_constraints = 18000


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
        # ae=0.2,
        L=30.0,
        gamma=3.2,
        Nx=Nx,
        Ny=32,
        Nz=32,
        Lx=Lx,
        Ly=1000,
        Lz=1000,
        aperiodic_x=False,
        aperiodic_y=True,
        aperiodic_z=True,
        parallel=True,
        corr_thres=0.0001,
        sinc_thres=12,
    )
    
    # print(stencil)