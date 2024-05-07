"""
Compares the computational time of serial and parallel turbulence generation
using Mannrs.
"""

from typing import Optional

from tqdm import trange
import numpy as np
import mannrs
import time

AE = 1
RES = 128


def run_single(
    resolution: int,
    N_stencil: Optional[int] = 1,
    N_turb: Optional[int] = 10,
    verbose: bool = False,
    parallel: bool = True,
):

    params = {
        "L": 1,
        "gamma": 3.2,
        "Lx": 100,
        "Ly": 100,
        "Lz": 100,
        "Nx": resolution,
        "Ny": resolution,
        "Nz": resolution,
    }

    stencil_times = []
    for _ in trange(max(N_stencil, 1), desc=" Generating stencils"):
        tstart = time.perf_counter()
        stencil = mannrs.Stencil(**params, parallel=parallel)
        stencil_times.append(time.perf_counter() - tstart)

    turb_times = []
    for seed in trange(max(N_turb, 1), desc=" Generating turbulence"):
        tstart = time.perf_counter()
        U, V, W = stencil.turbulence(AE, seed)
        turb_times.append(time.perf_counter() - tstart)

    return np.mean(stencil_times), np.mean(turb_times)


if __name__ == "__main__":

    t_stencil_ser, t_turb_ser = run_single(RES, N_stencil=3, parallel=False)
    print(
        f"SERIAL: stencil time: {t_stencil_ser:2.2f}s, turbulence generation time: {t_turb_ser:2.2f}s"
    )

    t_stencil_par, t_turb_par = run_single(RES, N_stencil=3)
    print(
        f"PARALLEL: stencil time: {t_stencil_par:2.2f}s, turbulence generation time: {t_turb_par:2.2f}s"
    )
    print(
        f"PARALLEL: stencil time: {t_stencil_ser /t_stencil_par:2.2f}x faster, turbulence generation time:   {t_turb_ser /t_turb_par:2.2f}x faster"
    )
