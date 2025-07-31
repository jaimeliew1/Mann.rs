"""
Example: Generates one or more unconstrained turbulence boxes using a shared stencil.
"""

import mannrs
from tqdm import trange


if __name__ == "__main__":
    # Parameters
    ae = 0.2
    N_boxes = 10

    print("Generating stencil...")
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

    print(f"Generating {N_boxes} turbulence boxes to turb/...")
    for seed in trange(N_boxes, desc="turbulence"):
        U, V, W = stencil.turbulence(ae, seed, domain="space", parallel=True)

        # mannrs.save_box(f"turb/U_{seed}.bin", U)
        # mannrs.save_box(f"turb/V_{seed}.bin", V)
        # mannrs.save_box(f"turb/W_{seed}.bin", W)
