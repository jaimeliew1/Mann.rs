"""
Plots the Mann frequency spectrum for given values of ae, L, and Gamma.
"""

import numpy as np
import mannrs
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Parameters
    ae = 1.0
    L = 33.6
    gamma = 3.9

    # Wavenumber range
    Kxs = np.logspace(-5, 1, 50)

    # Compute spectra
    UU, VV, WW, UW = mannrs.mann_spectra(Kxs, ae, L, gamma)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.semilogx(Kxs, Kxs * UU, "--", label="UU")
    plt.semilogx(Kxs, Kxs * VV, "--", label="VV")
    plt.semilogx(Kxs, Kxs * WW, "--", label="WW")
    plt.semilogx(Kxs, Kxs * UW, "--", label="UW")

    plt.title(rf"Mann Spectrum: $\gamma = {gamma:.2f}$")
    plt.xlabel("Wavenumber $k_1$ [rad/m]")
    plt.ylabel(r"Cross-spectra $k_1 \cdot \Phi(k_1)$ [(rad/m)(m²/s²)]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
