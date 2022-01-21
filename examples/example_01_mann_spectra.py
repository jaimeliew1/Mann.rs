import numpy as np
import rustmann
from tqdm import tqdm
import matplotlib.pyplot as plt
Lx, Ly, Lz = 1000, 1000, 1000
Ny, Nz = 33, 33
def one_comp_spec(kx, ae, L, gamma, sinc=False):

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
            if sinc:
                tensor = rustmann.sheared_tensor_sinc_f64(np.array([kx, ky, kz]), ae, L, gamma, Lx, Ly, Lz, Ny, Nz)
            else:
                tensor = rustmann.sheared_tensor_f64(np.array([kx, ky, kz]), ae, L, gamma)
            UU_grid[i, j] = r * tensor[0, 0]
            VV_grid[i, j] = r * tensor[1, 1]
            WW_grid[i, j] = r * tensor[2, 2]
            UW_grid[i, j] = r * tensor[0, 2]
    UU = np.trapz(np.trapz(UU_grid, Rs, axis=0), Thetas)
    VV = np.trapz(np.trapz(VV_grid, Rs, axis=0), Thetas)
    WW = np.trapz(np.trapz(WW_grid, Rs, axis=0), Thetas)
    UW = np.trapz(np.trapz(UW_grid, Rs, axis=0), Thetas)
    return UU, VV, WW, UW



ae = 1
L = 33.6
gamma = 3.9
if __name__ == "__main__":

    Kxs = np.logspace(-4, 2, 50)

    UU = np.zeros_like(Kxs)
    VV = np.zeros_like(Kxs)
    WW = np.zeros_like(Kxs)
    UW = np.zeros_like(Kxs)

    for j, kx in enumerate(tqdm(Kxs)):
        UU[j], VV[j], WW[j], UW[j] = one_comp_spec(kx, ae, L, gamma)

    plt.figure()
    plt.semilogx(Kxs, Kxs * UU, '--', label="UU (nosinc)")
    plt.semilogx(Kxs, Kxs * VV, '--', label="VV (nosinc)")
    plt.semilogx(Kxs, Kxs * WW, '--', label="WW (nosinc)")
    plt.semilogx(Kxs, Kxs * UW, '--', label="UW (nosinc)")
    UU = np.zeros_like(Kxs)
    VV = np.zeros_like(Kxs)
    WW = np.zeros_like(Kxs)
    UW = np.zeros_like(Kxs)

    # for j, kx in enumerate(tqdm(Kxs)):
    #     UU[j], VV[j], WW[j], UW[j] = one_comp_spec(kx, ae, L, gamma, sinc=True)

    # # plt.figure()
    # plt.semilogx(Kxs, Kxs * UU, label="UU")
    # plt.semilogx(Kxs, Kxs * VV, label="VV")
    # plt.semilogx(Kxs, Kxs * WW, label="WW")
    # plt.semilogx(Kxs, Kxs * UW, label="UW")

    plt.legend(loc="upper center", ncol=4)
    plt.grid()
    # plt.ylim(-1, 2.5)

    plt.title(rf"$\gamma={gamma:2.2f}$")
    plt.xlabel("Wave number, $k1$ [rad/m]")
    plt.ylabel("Cross spectra [(rad/m)(m^2/s^2)]")

    plt.show()
    # plt.savefig(fig_dir / f"spectra_gamma{gamma}.png", dpi=200, bbox_inches="tight")
