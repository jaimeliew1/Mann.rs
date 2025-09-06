"""
Constraint Turbulence Example

Generates constrained turbulence fields using sinusoidal velocity constraints
and visualizes the resulting flow patterns across multiple realizations.
"""

import matplotlib.pyplot as plt
import numpy as np

from mannrs.Stencil import Constraint, Stencil

# Domain and constraint parameters
Lx = 1000
Nx = 300
N_periods = 3
N_boxes = 10
N_constraints = 200


if __name__ == "__main__":
    # Generate constraints
    x_constraint = np.linspace(0, Lx, N_constraints)
    u_constraint = 10 * np.sin(N_periods * np.pi * x_constraint / Lx) + 5 * np.sin(
        5.2 * N_periods * np.pi * x_constraint / Lx
    )

    constraints = [
        Constraint(x=x, y=100, z=100, u=u) for x, u in zip(x_constraint, u_constraint)
    ]

    # Create constrained stencil
    print(f"Generating turbulence stencil with {len(constraints)} constraints...")
    stencil = (
        Stencil(
            L=30.0,
            gamma=3.2,
            Lx=Lx,
            Ly=200,
            Lz=200,
            Nx=Nx,
            Ny=32,
            Nz=32,
            sinc_thres=12.0,
            aperiodic_x=True,
        )
        .constrain(
            spectral_compression_target=0.8,
            constraints=constraints,
        )
        .build()
    )

    print(f"Correlation matrix sparsity: {100 * stencil.sparsity:.2f}%")
    print(f"Spectral compression: {100 * stencil.spectral_compression:.2f}%")

    # Generate multiple turbulence realizations
    U_slices = []
    print(f"Generating {N_boxes} constrained turbulence boxes...")
    for i in range(N_boxes):
        wf = stencil.turbulence(ae=0.2, seed=i)
        U_slices.append(wf.U[:, :, 16])

    # Plot results
    print("Plotting...")
    fig, axes = plt.subplots(1, N_boxes)

    for i, (slice, ax) in enumerate(zip(U_slices, axes)):
        ax.imshow(slice)
        ax.axis("off")
        ax.set_title(f"box {i + 1}")

    plt.figure()
    x = np.linspace(0, Lx, Nx)
    for slice in U_slices:
        plt.plot(x, slice[:, 16])
    plt.plot(x_constraint, u_constraint, ".k", label="constraints")
    plt.title("Centerline velocity")
    plt.xlabel("x [m]")
    plt.ylabel("U [m/s]")
    plt.legend()

    print("Done!")
    plt.show()
