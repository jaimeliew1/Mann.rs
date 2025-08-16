import mannrs

ae = 0.2
seed = 1234
params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000,
    "Ly": 200,
    "Lz": 200,
    "Nx": 32,
    "Ny": 32,
    "Nz": 32,
}


def test_mannrs():

    stencil = mannrs.Stencil(**params, parallel=False)

    U, V, W = stencil.turbulence(ae, seed)
    x, y, z = stencil.get_axes()
    assert len(x) == params["Nx"]
    assert len(y) == params["Ny"]
    assert len(z) == params["Nz"]



