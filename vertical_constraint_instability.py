import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mannrs.Stencil import Constraint, Stencil
from rich import print
from time import perf_counter


def make_vertical_constraints(Nv: int, seed: int, Nx: int = 350) -> list[Constraint]:
    rng = np.random.default_rng(seed=seed)

    py = 100.0
    pxs = np.linspace(0, 7000, Nx)
    constraints = []
    for pz in np.linspace(100, 200, Nv):
        constraints.extend(
            [Constraint(x=_x, y=py, z=pz, u=rng.normal(0, 5)) for _x in pxs]
        )
    return constraints

if __name__ == "__main__":
    constraints = make_vertical_constraints(Nv=5, seed=1234)
    # print(constraints)
    
    # Create constrained stencil
    stencil_def = Stencil(
        L=90.0,
        gamma=3.2,
        Lx=7000,
        Ly=200,
        Lz=210,
        Nx=300,
        Ny=32,
        Nz=32,
        sinc_thres=3.0,
        aperiodic_x=True,
        aperiodic_y=True,
        aperiodic_z=True,
    ).constrain(
        spectral_compression_target=0.9,
        constraints=constraints,
    )
    print(stencil_def)

    print(f"Generating turbulence stencil with {len(constraints)} constraints...")
    start_time = perf_counter()
    stencil = stencil_def.build(solver="lu")
    print(f"Stencil built in {perf_counter() - start_time:.2f} seconds.")

    kx, ky, kz, Suu, Svv, Sww, Suw = stencil.stencil.spectral_impulses()
    print(Suu.sum(), Svv.sum(), Sww.sum(), Suw.sum())
    print("generating turbulence...")
    start_time = perf_counter()
    wf = stencil.turbulence(ae=0.2, seed=1234)
    # wf2 = stencil.turbulence(ae=0.2, seed=1234)

    print(f"Turbulence generated in {perf_counter() - start_time:.2f} seconds.")

    # Plot a vertical and a horizontal slice
    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(wf.U[:, 16, :].T)
    axes[0].set_title("Vertical slice (y=16)")
    axes[1].imshow(wf.U[:, :, 16].T)
    axes[1].set_title("Horizontal slice (z=16)")
    plt.savefig("vertical_constraint_instability.png", dpi=300)

    # Plot a vertical and a horizontal slice


    # plot slices of spectral impulses
    fig, axes = plt.subplots(2, 2, figsize=(6, 12))
    axes[0, 0].imshow(
        Suu[:, :, 0].real.T,
        extent=(kx.min(), kx.max(), ky.min(), ky.max()),
        aspect="auto",
    )
    axes[0, 0].set_title("Suu slice (kz=0)")
    axes[0, 1].imshow(
        Svv[:, :, 0].real.T,
        extent=(kx.min(), kx.max(), ky.min(), ky.max()),
        aspect="auto",
    )
    axes[0, 1].set_title("Svv slice (kz=0)")
    axes[1, 0].imshow(
        Sww[:, :, 0].real.T,
        extent=(kx.min(), kx.max(), ky.min(), ky.max()),
        aspect="auto",
    )
    axes[1, 0].set_title("Sww slice (kz=0)")
    axes[1, 1].imshow(
        Suw[:, :, 0].real.T,
        extent=(kx.min(), kx.max(), ky.min(), ky.max()),
        aspect="auto",
    )
    axes[1, 1].set_title("Suw slice (kz=0)")
    plt.savefig("spectral_impulses_slices.png", dpi=300)
