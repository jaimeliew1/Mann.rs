"""
Generates multiple unconstrained turbulence boxes using a shared stencil. Plots
wind field slices of each box.
"""

import mannrs
from tqdm import trange
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameters
    ae = 0.2
    N_boxes = 10

    print("Generating turbulence stencil...")
    for _ in trange(1, desc="stencil"):
        stencil = mannrs.Stencil(
            L=30.0,
            gamma=3.2,
            Lx=6000,
            Ly=200,
            Lz=200,
            Nx=8192,
            Ny=32,
            Nz=32,
        )

    print(f"Generating {N_boxes} turbulence boxes...")
    U_slices = []
    for seed in trange(N_boxes, desc="turbulence"):
        U, V, W = stencil.turbulence(ae, seed)
        U_slices.append(U[0, :, :])

    print("Plotting...")
    fig, axes = plt.subplots(1, N_boxes, figsize=(16, 2))
    for i, (ax, slice) in enumerate(zip(axes, U_slices)):
        ax.imshow(slice)
        ax.axis("off")
        ax.set_title(f"box {i + 1}")

    plt.show()
