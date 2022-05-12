"""
Calculates the spectra of a generated turbulence box and compares it to the
theoretical spectra.
"""
from itertools import product
import numpy as np
from RustMann import Tensor, Stencil
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# Parameters from mann 1998 paper example
params = {
    "ae": 1,
    "L": 1,
    "gamma": 3.2,
    "Lx": 32,
    "Ly": 8,
    "Lz": 8,
    "Nx": 512,
    "Ny": 64,
    "Nz": 64,
}

ae = 1

c = [plt.cm.tab20(x / 10) for x in [0, 1, 2, 3]]
c_anal = [plt.cm.tab20(x / 10 + 1 / 20) for x in [0, 1, 2, 3]]


def one_comp_spec(kx, ae, L, gamma):

    tensor_gen = Tensor.Sheared(ae, L, gamma)
    Nr = 150
    Ntheta = 30
    Rs = np.logspace(-3, 7, Nr)
    Thetas = np.linspace(0, 2 * np.pi, Ntheta)

    UU_grid = np.zeros((Nr, Ntheta))
    VV_grid = np.zeros((Nr, Ntheta))
    WW_grid = np.zeros((Nr, Ntheta))
    UW_grid = np.zeros((Nr, Ntheta))
    for i, r in enumerate(Rs):
        for j, theta in enumerate(Thetas):
            ky = r * np.cos(theta)
            kz = r * np.sin(theta)

            tensor = tensor_gen.tensor([kx, ky, kz])
            UU_grid[i, j] = r * tensor[0, 0]
            VV_grid[i, j] = r * tensor[1, 1]
            WW_grid[i, j] = r * tensor[2, 2]
            UW_grid[i, j] = r * tensor[0, 2]
    UU = np.trapz(np.trapz(UU_grid, Rs, axis=0), Thetas)
    VV = np.trapz(np.trapz(VV_grid, Rs, axis=0), Thetas)
    WW = np.trapz(np.trapz(WW_grid, Rs, axis=0), Thetas)
    UW = np.trapz(np.trapz(UW_grid, Rs, axis=0), Thetas)
    return UU, VV, WW, UW


def get_spectra(U, V, W, Lmax, Nx):
    fs = 2 * np.pi / (Lmax / Nx)  # * 50

    f = np.fft.rfftfreq(Nx, 1 / fs)
    U_f = np.fft.rfft(U, axis=0)
    V_f = np.fft.rfft(V, axis=0)
    W_f = np.fft.rfft(W, axis=0)

    Suu = np.absolute(U_f) ** 2 / (fs * Nx)
    Svv = np.absolute(V_f) ** 2 / (fs * Nx)
    Sww = np.absolute(W_f) ** 2 / (fs * Nx)
    Suw = U_f * np.conj(W_f) / (fs * Nx)
    Suu = Suu.mean(axis=1).mean(axis=1)
    Svv = Svv.mean(axis=1).mean(axis=1)
    Sww = Sww.mean(axis=1).mean(axis=1)
    Suw = Suw.mean(axis=1).mean(axis=1)

    return f, Suu, Svv, Sww, Suw


def get_spectra_anal(ae, L, gamma):
    f_anal = np.logspace(-2, 1)

    UU = np.zeros_like(f_anal)
    VV = np.zeros_like(f_anal)
    WW = np.zeros_like(f_anal)
    UW = np.zeros_like(f_anal)

    for i, kx in enumerate(tqdm(f_anal)):
        UU[i], VV[i], WW[i], UW[i] = one_comp_spec(kx, ae, L, gamma)

    return f_anal, UU, VV, WW, UW


if __name__ == "__main__":

    stencil = Stencil(**params)

    Suu_list, Svv_list, Sww_list, Suw_list = [], [], [], []
    for seed in trange(20):
        U, V, W = stencil.turbulence(ae, seed)
        f, Suu, Svv, Sww, Suw = get_spectra(U, V, W, params["Lx"], params["Nx"])

        Suu_list.append(Suu)
        Svv_list.append(Svv)
        Sww_list.append(Sww)
        Suw_list.append(Suw)

    f_a, Suu_a, Svv_a, Sww_a, Suw_a = get_spectra_anal(
        params["ae"], params["L"], params["gamma"]
    )

    plt.figure()
    plt.axvline(3 / params["L"], lw=1, ls="--", c="k")
    plt.semilogx(f[1:], f[1:] * np.mean(Suu_list, axis=0)[1:], c=c[0], label="UU")
    plt.semilogx(f[1:], f[1:] * np.mean(Svv_list, axis=0)[1:], c=c[1], label="VV")
    plt.semilogx(f[1:], f[1:] * np.mean(Sww_list, axis=0)[1:], c=c[2], label="WW")
    plt.semilogx(f[1:], f[1:] * np.mean(Suw_list, axis=0)[1:].real, c=c[3], label="UW")

    plt.semilogx(f_a, f_a * Suu_a, "--", c=c_anal[0], label="UU_a")
    plt.semilogx(f_a, f_a * Svv_a, "--", c=c_anal[1], label="VV_a")
    plt.semilogx(f_a, f_a * Sww_a, "--", c=c_anal[2], label="WW_a")
    plt.semilogx(f_a, f_a * Suw_a.real, "--", c=c_anal[3], label="UW_a")

    plt.legend(ncol=2, fontsize="x-small")
    plt.grid()
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")
    plt.savefig("spectra.png", dpi=200, bbox_inches="tight")
    plt.show()
