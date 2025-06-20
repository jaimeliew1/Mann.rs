"""
Plots the Mann frequency spectrum for given values of ae, L, and Gamma.
"""

import numpy as np
from mannrs import mann_spectra
import matplotlib.pyplot as plt


ae = 1
L = 33.6
gamma = 3.9


if __name__ == "__main__":

    Kxs = np.logspace(-5, 1, 50)

    UU, VV, WW, UW = mann_spectra(Kxs, ae, L, gamma)
    plt.figure()

    plt.semilogx(Kxs, Kxs * UU, "--", label="UU")
    plt.semilogx(Kxs, Kxs * VV, "--", label="VV")
    plt.semilogx(Kxs, Kxs * WW, "--", label="WW")
    plt.semilogx(Kxs, Kxs * UW, "--", label="UW")

    plt.legend()
    plt.grid()

    plt.title(rf"$\gamma={gamma:2.2f}$")
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")

    plt.show()
