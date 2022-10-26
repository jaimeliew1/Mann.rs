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
    "Ny": 64,
    "Nz": 64,
}

N = 10

if __name__ == "__main__":
    print("Generating stencil...")
    for _ in trange(1, desc="stencil"):
        stencil = mannrs.Stencil(**params, parallel=True)

    print(f"Generating {N} turbulence boxes to turb/...")
    for seed in trange(N, desc="turbulence"):
        U, V, W = stencil.turbulence(ae, seed, domain="space")



        mannrs.save_box(f"turb/U_{seed}.bin", U)
        mannrs.save_box(f"turb/V_{seed}.bin", V)
        mannrs.save_box(f"turb/W_{seed}.bin", W)
