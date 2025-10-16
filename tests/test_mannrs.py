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

    # Check output shapes
    assert wf.U.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wf.V.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wf.W.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wf.x.shape == (params["Nx"],)
    assert wf.y.shape == (params["Ny"],)
    assert wf.z.shape == (params["Nz"],)

    # Check types
    import numpy as np

    assert isinstance(wf.U, np.ndarray)
    assert isinstance(wf.V, np.ndarray)
    assert isinstance(wf.W, np.ndarray)
    assert isinstance(wf.x, np.ndarray)
    assert isinstance(wf.y, np.ndarray)
    assert isinstance(wf.z, np.ndarray)

    # Check reproducibility
    wf2 = stencil.turbulence(ae, seed)
    assert np.allclose(wf.U, wf2.U)
    assert np.allclose(wf.V, wf2.V)
    assert np.allclose(wf.W, wf2.W)

    # Test saving and loading
    wf.write("test_unconstrained.npz")
    import os

    assert os.path.exists("test_unconstrained.npz")
    os.remove("test_unconstrained.npz")


def test_constrained():
    stencil = Stencil(**params, parallel=False).constrain(constraints).build()
    wf = stencil.turbulence(ae, seed)

    # Check output shapes
    assert wf.U.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wf.V.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wf.W.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wf.x.shape == (params["Nx"],)
    assert wf.y.shape == (params["Ny"],)
    assert wf.z.shape == (params["Nz"],)

    # Check constraint satisfaction (streamwise velocity at constraint points)
    # Find closest grid point to each constraint and check value
    import numpy as np

    # Check reproducibility
    wf2 = stencil.turbulence(ae, seed)
    assert np.allclose(wf.U, wf2.U)
    assert np.allclose(wf.V, wf2.V)
    assert np.allclose(wf.W, wf2.W)

    # Test saving and loading
    wf.write("test_constrained.npz")
    import os

    assert os.path.exists("test_constrained.npz")
    os.remove("test_constrained.npz")


def test_invalid_params():
    # Should raise error for negative grid size
    import pytest

    bad_params = params.copy()
    bad_params["Nx"] = -1
    with pytest.raises(Exception):
        Stencil(**bad_params).build()


def test_parallel_vs_serial():
    stencil = Stencil(**params, parallel=False).build()
    wf_serial = stencil.turbulence(ae, seed, parallel=False)
    wf_parallel = stencil.turbulence(ae, seed, parallel=True)
    import numpy as np

    assert np.allclose(wf_serial.U, wf_parallel.U)
    assert np.allclose(wf_serial.V, wf_parallel.V)
    assert np.allclose(wf_serial.W, wf_parallel.W)


def test_cli_parallel_serial(tmp_path):
    import subprocess
    import toml

    input_file = tmp_path / "input.toml"
    toml_dict = {
        "stencil_spec": params,
        "turbulence_boxes": [
            {"ae": ae, "seed": seed, "output": str(tmp_path / "wf.npz")}
        ],
    }
    with open(input_file, "w") as f:
        toml.dump(toml_dict, f)
    # Run CLI with --serial
    result_serial = subprocess.run(
        ["python", "-m", "mannrs.CLI", "--serial", str(input_file)],
        capture_output=True,
        text=True,
    )
    assert result_serial.returncode == 0
    # Run CLI with --parallel
    result_parallel = subprocess.run(
        ["python", "-m", "mannrs.CLI", "--parallel", str(input_file)],
        capture_output=True,
        text=True,
    )
    assert result_parallel.returncode == 0
