"""
Generates several turbulence boxes from a single stencil.
"""
import mannrs
from tqdm import trange

ae = 0.2
params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000,
    "Ly": 200,
    "Lz": 200,
    "Nx": 8192,
    "Ny": 32,
    "Nz": 32,
}

N = 1

if __name__ == "__main__":
    print("Generating stencil...")
    for _ in trange(1, desc="stencil"):
        stencil = mannrs.Stencil(**params, parallel=True, aperiodic_x=True, aperiodic_y=True, aperiodic_z=True)

    print(f"Generating {N} turbulence boxes to turb/...")
    for seed in trange(N, desc="turbulence"):
        U, V, W = stencil.turbulence(ae, seed, domain="space", parallel=True)
        print(U.shape)


        # mannrs.save_box(f"turb/U_{seed}.bin", U)
        # mannrs.save_box(f"turb/V_{seed}.bin", V)
        # mannrs.save_box(f"turb/W_{seed}.bin", W)
