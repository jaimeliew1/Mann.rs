"""
Plots the Mann frequency spectrum for given values of ae, L, and Gamma. Comparse
time of Rust and Python implementations.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import mannrs
from mannrs import Tensor

ae = 1
L = 33.6
gamma = 3.9


def one_comp_spec(kx, ae, L, gamma):
    tensor_gen = Tensor.Sheared(ae, L, gamma)
    Nr = 150
    Ntheta = 30
    Rs = np.logspace(-4, 7, Nr)
    Thetas = np.linspace(0, 2 * np.pi, Ntheta)

    UU_grid = np.zeros((Nr, Ntheta))
    VV_grid = np.zeros((Nr, Ntheta))
    WW_grid = np.zeros((Nr, Ntheta))
    UW_grid = np.zeros((Nr, Ntheta))
    for i, r in enumerate(Rs):
        for j, theta in enumerate(Thetas):
            ky = r * np.cos(theta)
            kz = r * np.sin(theta)

            tensor = tensor_gen.tensor(np.array([kx, ky, kz]))
            UU_grid[i, j] = r * tensor[0, 0]
            VV_grid[i, j] = r * tensor[1, 1]
            WW_grid[i, j] = r * tensor[2, 2]
            UW_grid[i, j] = r * tensor[0, 2]
    UU = np.trapezoid(np.trapezoid(UU_grid, Rs, axis=0), Thetas)
    VV = np.trapezoid(np.trapezoid(VV_grid, Rs, axis=0), Thetas)
    WW = np.trapezoid(np.trapezoid(WW_grid, Rs, axis=0), Thetas)
    UW = np.trapezoid(np.trapezoid(UW_grid, Rs, axis=0), Thetas)
    return UU, VV, WW, UW


if __name__ == "__main__":
    Kxs = np.logspace(-5, 2, 100)

    # Calculate Mann spectra in Rust
    t_start = time.perf_counter()
    _Suu, _Svv, _Sww, _Suw = mannrs.spectra(Kxs, ae, L, gamma)
    print(f"Time elapsed (rust): {time.perf_counter() - t_start:.2f} seconds")
    t_start = time.perf_counter()
    # Calculate Mann spectra in Python
    UU = np.zeros_like(Kxs)
    VV = np.zeros_like(Kxs)
    WW = np.zeros_like(Kxs)
    UW = np.zeros_like(Kxs)

    for j, kx in enumerate(Kxs):
        UU[j], VV[j], WW[j], UW[j] = one_comp_spec(kx, ae, L, gamma)
    print(f"Time elapsed (Python): {time.perf_counter() - t_start:.2f} seconds")

    # plot
    plt.figure()
    plt.semilogx(Kxs, Kxs * UU, "--", label="UU (Python)")
    plt.semilogx(Kxs, Kxs * VV, "--", label="VV (Python)")
    plt.semilogx(Kxs, Kxs * WW, "--", label="WW (Python)")
    plt.semilogx(Kxs, Kxs * UW, "--", label="UW (Python)")

    plt.semilogx(Kxs, Kxs * _Suu, ":", label="UU (Rust)")
    plt.semilogx(Kxs, Kxs * _Svv, ":", label="VV (Rust)")
    plt.semilogx(Kxs, Kxs * _Sww, ":", label="WW (Rust)")
    plt.semilogx(Kxs, Kxs * _Suw, ":", label="UW (Rust)")

    plt.legend()
    plt.grid()

    plt.title(rf"$\gamma={gamma:2.2f}$")
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")

    plt.show()
    plt.savefig("qwe.png")