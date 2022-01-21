import numpy as np
import rustmann
from tqdm import trange
import time


Nx, Ny, Nz = 8192, 256, 256
Lx, Ly, Lz = 8192, 200, 200
ae = 0.2
gamma = 3.2
L = 30.0
seed = 1000


if __name__ == "__main__":

    t_start = time.time()
    print("Generating stencil...", end="")

    stencil = rustmann.stencilate_f64(ae, L, gamma, Lx, Ly, Lz, Nx, Ny, Nz)
    print(f"Done ({time.time() - t_start:2.2f} seconds)")

    t_start = time.time()
    print("Generating stencil...", end="")

    U, V, W = rustmann.turbulate_f64(stencil, seed, Nx, Ny, Nz, Lx, Ly, Lz)

    print(f"Done ({time.time() - t_start:2.2f} seconds)")
