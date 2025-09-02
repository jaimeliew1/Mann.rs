"""
Generates a synthetic turbulence field and compares its computed power spectra
against theoretical Mann spectra.
"""

import numpy as np
from mannrs import Stencil, mann_spectra
from tqdm import trange
import matplotlib.pyplot as plt

# Parameters from Mann 1998 paper example
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


def calculate_box_spectra(
    U: np.ndarray, V: np.ndarray, W: np.ndarray, Lx: float, Nx: int
) -> tuple[np.ndarray, ...]:
    """
    Calculate power spectra and cross-spectra from 3D wind field components.

    Parameters:
    -----------
    U, V, W : ndarray
        3D wind velocity components with shape (Nx, ny, nz)
        where the first axis is the spatial direction for spectral analysis
    Lx : float
        Length of the domain in the spectral analysis direction
    Nx : int
        Number of grid points along the spectral analysis direction

    Returns:
    --------
    frequencies : ndarray
        Frequency array for the spectra
    Suu, Svv, Sww : ndarray
        Auto-power spectra for U, V, W components
    Suw : ndarray
        Cross-power spectrum between U and W components
    """
    # Calculate sampling frequency and frequency array
    sampling_freq = 2 * np.pi / (Lx / Nx)
    frequencies = np.fft.rfftfreq(Nx, 1 / sampling_freq)

    # Compute FFTs along the first axis (spectral direction)
    U_fft = np.fft.rfft(U, axis=0)
    V_fft = np.fft.rfft(V, axis=0)
    W_fft = np.fft.rfft(W, axis=0)

    # Calculate power spectral densities
    normalization = sampling_freq * Nx

    Suu = np.abs(U_fft) ** 2 / normalization
    Svv = np.abs(V_fft) ** 2 / normalization
    Sww = np.abs(W_fft) ** 2 / normalization

    # Calculate cross-power spectral density
    Suw = U_fft * np.conj(W_fft) / normalization

    # Average over the lateral dimensions (y and z directions)
    Suu = np.mean(Suu, axis=(1, 2))
    Svv = np.mean(Svv, axis=(1, 2))
    Sww = np.mean(Sww, axis=(1, 2))
    Suw = np.mean(Suw, axis=(1, 2))

    return frequencies, Suu, Svv, Sww, Suw


if __name__ == "__main__":
    stencil = Stencil(**params)

    Suu_list, Svv_list, Sww_list, Suw_list = [], [], [], []
    for seed in trange(20):
        wf = stencil.turbulence(ae, seed)
        f, Suu, Svv, Sww, Suw = calculate_box_spectra(
            wf.U, wf.V, wf.W, params["Lx"], params["Nx"]
        )

        Suu_list.append(Suu)
        Svv_list.append(Svv)
        Sww_list.append(Sww)
        Suw_list.append(Suw)

    f_a = np.logspace(-2, 1)
    Suu_a, Svv_a, Sww_a, Suw_a = mann_spectra(f_a, ae, params["L"], params["gamma"])

    c = [plt.cm.tab20(x / 10) for x in [0, 1, 2, 3]]
    c_anal = [plt.cm.tab20(x / 10 + 1 / 20) for x in [0, 1, 2, 3]]

    plt.figure()
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
    plt.show()
