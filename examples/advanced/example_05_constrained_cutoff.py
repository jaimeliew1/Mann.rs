import matplotlib.pyplot as plt
import numpy as np

from mannrs import ConstrainedStencil, Constraint, Stencil

params = {}
Lx = 1000
Ly = 200
Lz = 200
Nx = 300
Ny = 32
Nz = 32
THRES = 0.000001

if __name__ == "__main__":
    x_constraint = np.arange(0, 2000, 2)
    u_constraint = 10 * np.sin(x_constraint / 10)
    constraints = [Constraint(x, 60, 0, u) for x, u in zip(x_constraint, u_constraint)]

    stencil = Stencil(
        # constraints=constraints,
        L=30.0,
        gamma=3.2,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        aperiodic_x=True,
        aperiodic_y=True,
        aperiodic_z=True,
        parallel=True,
    )
    print(stencil.stencil)

    kxs = np.fft.fftfreq(2 * Nx, Lx / Nx)
    kys = np.fft.fftfreq(2 * Ny, Ly / Ny)
    kzs = np.fft.rfftfreq(2 * Nz, Lz / Nz)
    RUU_f, RVV_f, RWW_f, RUW_f = stencil.stencil.spectral_component_grids()

    xroll = len(kxs) // 2
    yroll = len(kys) // 2
    zroll = 0

    print(xroll, yroll, zroll)

    kxs = np.roll(kxs, xroll, axis=0)
    kys = np.roll(kys, yroll, axis=0)
    kzs = np.roll(kzs, zroll, axis=0)

    RUU_f = np.roll(RUU_f, (xroll, yroll, zroll), (0, 1, 2))
    RVV_f = np.roll(RVV_f, (xroll, yroll, zroll), (0, 1, 2))
    RWW_f = np.roll(RWW_f, (xroll, yroll, zroll), (0, 1, 2))
    RUW_f = np.roll(RUW_f, (xroll, yroll, zroll), (0, 1, 2))

    RUU_f_max, RVV_f_max, RWW_f_max = RUU_f.max(), RVV_f.max(), RWW_f.max()
    ind_x_min = np.where(RUU_f[:, yroll, zroll] >= THRES * RUU_f_max)[0][0]
    ind_x_max = len(kxs) - np.where(RUU_f[:, yroll, zroll][::-1] >= THRES * RUU_f_max)[0][0]

    ind_y_min = np.where(RVV_f[xroll, :, zroll] >= THRES * RVV_f_max)[0][0]
    ind_y_max = len(kys) - np.where(RVV_f[xroll, :, zroll][::-1] >= THRES * RVV_f_max)[0][0]
    
    ind_z_max = np.where(RWW_f[xroll, yroll, :] < THRES * RWW_f_max)[0][0]
    print(f"ind_x_min: {ind_x_min}")
    print(f"ind_x_max: {ind_x_max}")
    print(f"ind_y_min: {ind_y_min}")
    print(f"ind_y_max: {ind_y_max}")
    print(f"ind_z_max: {ind_z_max}")

    print(RUU_f.shape)
    print(RUU_f.sum())
    print(RUU_f[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max].sum())

    fig, axes = plt.subplots(4, 1)

    axes[0].plot(kxs, RUU_f[:, 0, 0])
    axes[0].plot(kxs, RVV_f[:, 0, 0])
    axes[0].plot(kxs, RWW_f[:, 0, 0])

    axes[1].plot(kys, RUU_f[0, :, 0])
    axes[1].plot(kys, RVV_f[0, :, 0])
    axes[1].plot(kys, RWW_f[0, :, 0])

    axes[2].plot(kzs, RUU_f[0, 0, :])
    axes[2].plot(kzs, RVV_f[0, 0, :])
    axes[2].plot(kzs, RWW_f[0, 0, :])

    axes[3].plot(kxs)

    plt.savefig("R_f.png", dpi=300, bbox_inches="tight")

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(RUU_f[ind_x_min:ind_x_max, ind_y_min:ind_y_max, 0], vmin=0, vmax=RUU_f.max())
    axes[1].imshow(RUU_f[ind_x_min:ind_x_max, 32, :ind_z_max], vmin=0, vmax=RUU_f.max())
    axes[2].imshow(RUU_f[300, ind_y_min:ind_y_max, :ind_z_max], vmin=0, vmax=RUU_f.max())
    plt.savefig("R_f_slice.png", dpi=300, bbox_inches="tight")
    # plt.show()
