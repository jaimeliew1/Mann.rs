"""
Compares the computational time of serial and parallel turbulence generation
using RustMann.
"""
import RustMann
import time

ae = 1
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


if __name__ == "__main__":
    print("Generating stencil (serial)...", end=" ")
    tstart = time.time()
    stencil_serial = RustMann.Stencil(**params)
    t_stencil_serial = time.time() - tstart
    print(f"{t_stencil_serial:2.1f}s")

    print("Generating stencil (parallel)...", end="")
    tstart = time.time()
    stencil_par = RustMann.Stencil(**params, parallel=True)
    t_stencil_par = time.time() - tstart
    print(f"{t_stencil_par:2.1f}s")

    print("\nGenerating turbulence (serial)...", end=" ")
    tstart = time.time()
    U, V, W = stencil_par.turbulence(ae, 1)
    t_turb_serial = time.time() - tstart
    print(f"{t_turb_serial:2.1f}s")

    print("Generating turbulence (parallel)...", end=" ")
    tstart = time.time()
    U_par, V_par, W_par = stencil_par.turbulence(ae, 1, parallel=True)
    t_turb_par = time.time() - tstart
    print(f"{t_turb_par:2.1f}s")

    # Assert that serial and parallel results are equal.
    assert ((U - U_par) ** 2).sum() < 1e-16
    assert ((V - V_par) ** 2).sum() < 1e-16
    assert ((W - W_par) ** 2).sum() < 1e-16
