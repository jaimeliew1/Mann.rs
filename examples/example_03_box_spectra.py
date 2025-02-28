"""
Calculates the spectra of a generated turbulence box and compares it to the
theoretical spectra.
"""

import numpy as np
from mannrs import Stencil, spectra
from tqdm import trange
import matplotlib.pyplot as plt

# Parameters from mann 1998 paper example
params = {
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


def get_spectra(U, V, W, Lmax, Nx):
    fs = 2 * np.pi / (Lmax / Nx)

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

    Suu_a, Svv_a, Sww_a, Suw_a = spectra(f, ae, params["L"], params["gamma"])

    plt.figure()
    plt.axvline(3 / params["L"], lw=1, ls="--", c="k")
    plt.semilogx(f[1:], f[1:] * np.mean(Suu_list, axis=0)[1:], c=c[0], label="UU")
    plt.semilogx(f[1:], f[1:] * np.mean(Svv_list, axis=0)[1:], c=c[1], label="VV")
    plt.semilogx(f[1:], f[1:] * np.mean(Sww_list, axis=0)[1:], c=c[2], label="WW")
    plt.semilogx(f[1:], f[1:] * np.mean(Suw_list, axis=0)[1:].real, c=c[3], label="UW")

    plt.semilogx(f, f * Suu_a, "--", c=c_anal[0], label="UU_a")
    plt.semilogx(f, f * Svv_a, "--", c=c_anal[1], label="VV_a")
    plt.semilogx(f, f * Sww_a, "--", c=c_anal[2], label="WW_a")
    plt.semilogx(f, f * Suw_a.real, "--", c=c_anal[3], label="UW_a")

    plt.legend(ncol=2, fontsize="x-small")
    plt.grid()
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")
    plt.savefig("spectra.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.savefig("asdf.png")
