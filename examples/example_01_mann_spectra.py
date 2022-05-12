"""
Plots the Mann frequency spectrum for given values of ae, L, and Gamma.
"""
import numpy as np
from RustMann import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    UU = np.trapz(np.trapz(UU_grid, Rs, axis=0), Thetas)
    VV = np.trapz(np.trapz(VV_grid, Rs, axis=0), Thetas)
    WW = np.trapz(np.trapz(WW_grid, Rs, axis=0), Thetas)
    UW = np.trapz(np.trapz(UW_grid, Rs, axis=0), Thetas)
    return UU, VV, WW, UW


if __name__ == "__main__":

    Kxs = np.logspace(-5, 1, 50)

    UU = np.zeros_like(Kxs)
    VV = np.zeros_like(Kxs)
    WW = np.zeros_like(Kxs)
    UW = np.zeros_like(Kxs)

    for j, kx in enumerate(tqdm(Kxs)):
        UU[j], VV[j], WW[j], UW[j] = one_comp_spec(kx, ae, L, gamma)

    plt.figure()
    plt.semilogx(Kxs, Kxs * UU, "--", label="UU")
    plt.semilogx(Kxs, Kxs * VV, "--", label="VV")
    plt.semilogx(Kxs, Kxs * WW, "--", label="WW")
    plt.semilogx(Kxs, Kxs * UW, "--", label="UW")
    UU = np.zeros_like(Kxs)
    VV = np.zeros_like(Kxs)
    WW = np.zeros_like(Kxs)
    UW = np.zeros_like(Kxs)

    plt.legend()
    plt.grid()

    plt.title(rf"$\gamma={gamma:2.2f}$")
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")

    plt.show()
