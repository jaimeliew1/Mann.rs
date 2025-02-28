"""
Plots the Mann frequency spectrum for given values of ae, L, and Gamma.
"""

import matplotlib.pyplot as plt
import numpy as np

import mannrs

ae = 1
L = 33.6
gamma = 3.9


if __name__ == "__main__":
    Kxs = np.logspace(-5, 1, 50)

    Suu, Svv, Sww, Suw = mannrs.spectra(Kxs, ae, L, gamma)

    plt.figure()

    plt.semilogx(Kxs, Kxs * Suu, "--", label="UU")
    plt.semilogx(Kxs, Kxs * Svv, "--", label="VV")
    plt.semilogx(Kxs, Kxs * Sww, "--", label="WW")
    plt.semilogx(Kxs, Kxs * Suw, "--", label="UW")

    plt.legend()
    plt.grid()

    plt.title(rf"$\gamma={gamma:2.2f}$")
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")

    plt.show()
    plt.savefig("asfd.png")
