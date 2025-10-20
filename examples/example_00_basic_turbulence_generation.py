from mannrs import Stencil


mann_params = {
    "L": 30.0,  # Length scale
    "gamma": 3.2,  # Anisotropy parameter
    "Lx": 6000.0,  # domain size in x
    "Ly": 200.0,  # domain size in y
    "Lz": 200.0,  # domain size in z
    "Nx": 8192,  # number of grid points in x
    "Ny": 32,  # number of grid points in y
    "Nz": 32,  # number of grid points in z
}

# turbulence amplitude and random seed
ae, seed = 0.2, 1234
filename = "output.npz"


if __name__ == "__main__":
    # Build the stencil, generate turbulence, and write to disk in one chained call.
    Stencil(**mann_params).build().turbulence(ae, seed).write(filename)
