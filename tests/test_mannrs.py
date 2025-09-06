from mannrs import Constraint, Stencil

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

constraints = [
    Constraint(x=100, y=100, z=100, u=0.5),
    Constraint(x=150, y=100, z=100, u=-0.5),
    Constraint(x=200, y=100, z=100, u=0.0),
]


def test_unconstrained():
    stencil = Stencil(**params, parallel=False).build()
    wf = stencil.turbulence(ae, seed)

    assert len(wf.x) == params["Nx"]
    assert len(wf.y) == params["Ny"]
    assert len(wf.z) == params["Nz"]


def test_constrained():
    stencil = Stencil(**params, parallel=False).constrain(constraints).build()
    wf = stencil.turbulence(ae, seed)

    assert len(wf.x) == params["Nx"]
    assert len(wf.y) == params["Ny"]
    assert len(wf.z) == params["Nz"]
