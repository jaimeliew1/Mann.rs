import matplotlib.pyplot as plt
import numpy as np

from mannrs import ConstrainedStencil, Constraint

params = {}
Lx = 1000
Ly = 200
Lz = 200
Nx = 300
Ny = 32
Nz = 32


if __name__ == "__main__":
    x_constraint = np.arange(0, 2000, 2)
    u_constraint = 10 * np.sin(x_constraint / 10)
    constraints = [Constraint(x, 60, 0, u) for x, u in zip(x_constraint, u_constraint)]

    stencil = ConstrainedStencil(
            constraints=constraints,
            ae=0.2,
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
    RUU_f, RVV_f, RWW_f, RUW_f = stencil.stencil.stencil.spectral_component_grids()

    print(RUU_f.shape)
    print(RUU_f[0,0,0])

    fig, axes = plt.subplots(4, 1)
    
    axes[0].plot(kxs, RUU_f[:,0,0])
    axes[0].plot(kxs, RVV_f[:,0,0])
    axes[0].plot(kxs, RWW_f[:,0,0])

    axes[1].plot(kys, RUU_f[0,:,0])
    axes[1].plot(kys, RVV_f[0,:,0])
    axes[1].plot(kys, RWW_f[0,:,0])

    axes[2].plot(kzs, RUU_f[0,0,:])
    axes[2].plot(kzs, RVV_f[0,0,:])
    axes[2].plot(kzs, RWW_f[0,0,:])

    axes[3].plot(kxs)


    plt.savefig("R_f.png", dpi=300, bbox_inches="tight")