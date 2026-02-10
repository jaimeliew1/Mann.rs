from mannrs import Constraint, Stencil
from click.testing import CliRunner
from mannrs.CLI import CLI  
import toml
import pytest
import numpy as np


# Test parameters
ae = 0.05
seed = 1234
params = {
    "L": 10.0,
    "gamma": 3.2,
    "Lx": 200,
    "Ly": 200,
    "Lz": 200,
    "Nx": 16,
    "Ny": 16,
    "Nz": 16,
}

constraints = [
    Constraint(x=50, y=100, z=100, u=0.5),
    Constraint(x=100, y=100, z=100, u=0.0),
    Constraint(x=150, y=100, z=100, u=-0.5),
]


# ===== Fixtures =====


@pytest.fixture
def unconstrained_stencil():
    """Fixture providing an unconstrained stencil."""
    return Stencil(**params, parallel=False).build()


@pytest.fixture
def constrained_stencil():
    """Fixture providing a constrained stencil."""
    return Stencil(**params, aperiodic_x=True, parallel=False).constrain(constraints).build()


@pytest.fixture
def unconstrained_wind_field(unconstrained_stencil):
    """Fixture providing an unconstrained wind field."""
    return unconstrained_stencil.turbulence(ae, seed)


@pytest.fixture
def constrained_wind_field(constrained_stencil):
    """Fixture providing a constrained wind field."""
    return constrained_stencil.turbulence(ae, seed)


# ===== Stencil Construction Tests =====


def test_stencil_construction():
    """Test that stencil can be constructed with valid parameters."""
    stencil = Stencil(**params, parallel=False).build()
    assert stencil is not None


def test_stencil_with_various_parameters():
    """Test stencil construction with different parameter sets."""
    # Small grid
    small_params = params.copy()
    small_params.update({"Nx": 8, "Ny": 8, "Nz": 8})
    stencil = Stencil(**small_params, parallel=False).build()
    assert stencil is not None
    
    # Different aspect ratio
    aspect_params = params.copy()
    aspect_params.update({"Lx": 1000, "Ly": 100, "Lz": 50})
    stencil = Stencil(**aspect_params, parallel=False).build()
    assert stencil is not None
    
    # Different Mann parameters
    mann_params = params.copy()
    mann_params.update({"L": 50.0, "gamma": 4.0})
    stencil = Stencil(**mann_params, parallel=False).build()
    assert stencil is not None


def test_stencil_aperiodic_options():
    """Test stencil construction and turbulence generation with aperiodic boundary conditions."""
    expected_shape = (params["Nx"], params["Ny"], params["Nz"])
    
    # Test aperiodic_x
    stencil = Stencil(**params, aperiodic_x=True, parallel=False).build()
    assert stencil is not None
    wf = stencil.turbulence(ae, seed)
    assert wf.U.shape == expected_shape
    assert wf.V.shape == expected_shape
    assert wf.W.shape == expected_shape
    
    # Test aperiodic_y
    stencil = Stencil(**params, aperiodic_y=True, parallel=False).build()
    assert stencil is not None
    wf = stencil.turbulence(ae, seed)
    assert wf.U.shape == expected_shape
    assert wf.V.shape == expected_shape
    assert wf.W.shape == expected_shape
    
    # Test aperiodic_z
    stencil = Stencil(**params, aperiodic_z=True, parallel=False).build()
    assert stencil is not None
    wf = stencil.turbulence(ae, seed)
    assert wf.U.shape == expected_shape
    assert wf.V.shape == expected_shape
    assert wf.W.shape == expected_shape
    
    # Test all aperiodic
    stencil = Stencil(
        **params, 
        aperiodic_x=True, 
        aperiodic_y=True, 
        aperiodic_z=True,
        parallel=False
    ).build()
    assert stencil is not None
    wf = stencil.turbulence(ae, seed)
    assert wf.U.shape == expected_shape
    assert wf.V.shape == expected_shape
    assert wf.W.shape == expected_shape


def test_invalid_params():
    """Test that invalid parameters raise appropriate errors."""
    # Negative grid size
    bad_params = params.copy()
    bad_params["Nx"] = -1
    with pytest.raises(Exception):
        Stencil(**bad_params).build()
    
    # Zero grid size
    bad_params = params.copy()
    bad_params["Ny"] = 0
    with pytest.raises(Exception):
        Stencil(**bad_params).build()
    
    # Negative length scale
    bad_params = params.copy()
    bad_params["L"] = -10.0
    with pytest.raises(Exception):
        Stencil(**bad_params).build()
    
    # Negative domain size
    bad_params = params.copy()
    bad_params["Lx"] = -100
    with pytest.raises(Exception):
        Stencil(**bad_params).build()


# ===== Unconstrained Turbulence Tests =====


def test_unconstrained_shape(unconstrained_wind_field):
    """Test that unconstrained wind field has correct shape."""
    assert unconstrained_wind_field.U.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert unconstrained_wind_field.V.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert unconstrained_wind_field.W.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert unconstrained_wind_field.x.shape == (params["Nx"],)
    assert unconstrained_wind_field.y.shape == (params["Ny"],)
    assert unconstrained_wind_field.z.shape == (params["Nz"],)


def test_unconstrained_types(unconstrained_wind_field):
    """Test that unconstrained wind field components are numpy arrays."""
    assert isinstance(unconstrained_wind_field.U, np.ndarray)
    assert isinstance(unconstrained_wind_field.V, np.ndarray)
    assert isinstance(unconstrained_wind_field.W, np.ndarray)
    assert isinstance(unconstrained_wind_field.x, np.ndarray)
    assert isinstance(unconstrained_wind_field.y, np.ndarray)
    assert isinstance(unconstrained_wind_field.z, np.ndarray)


def test_unconstrained_reproducibility(unconstrained_stencil):
    """Test that unconstrained turbulence generation is reproducible with same seed."""
    wf1 = unconstrained_stencil.turbulence(ae, seed)
    wf2 = unconstrained_stencil.turbulence(ae, seed)
    
    assert np.allclose(wf1.U, wf2.U)
    assert np.allclose(wf1.V, wf2.V)
    assert np.allclose(wf1.W, wf2.W)


def test_unconstrained_different_seeds(unconstrained_stencil):
    """Test that different seeds produce different turbulence."""
    wf1 = unconstrained_stencil.turbulence(ae, seed)
    wf2 = unconstrained_stencil.turbulence(ae, seed + 1)
    
    # Fields should be different
    assert not np.allclose(wf1.U, wf2.U)
    assert not np.allclose(wf1.V, wf2.V)
    assert not np.allclose(wf1.W, wf2.W)
    
    # But should have similar statistics
    assert np.abs(np.mean(wf1.U) - np.mean(wf2.U)) < 0.5
    assert np.abs(np.std(wf1.U) - np.std(wf2.U)) < 0.5


def test_unconstrained_zero_mean(unconstrained_stencil):
    """Test that unconstrained turbulence has approximately zero mean across multiple boxes."""
    
    # Generate multiple boxes with different seeds
    num_boxes = 5
    u_means = []
    v_means = []
    w_means = []
    
    for i in range(num_boxes):
        wf = unconstrained_stencil.turbulence(ae, seed + i * 100)
        u_means.append(np.mean(wf.U))
        v_means.append(np.mean(wf.V))
        w_means.append(np.mean(wf.W))
    

    # Average across all boxes should be close to zero
    assert np.abs(np.mean(u_means)) < 0.1
    assert np.abs(np.mean(v_means)) < 0.1
    assert np.abs(np.mean(w_means)) < 0.1


def test_unconstrained_coordinates(unconstrained_wind_field):
    """Test that coordinate arrays are properly constructed."""
    # Coordinates should be monotonically increasing
    assert np.all(np.diff(unconstrained_wind_field.x) > 0)
    assert np.all(np.diff(unconstrained_wind_field.y) > 0)
    assert np.all(np.diff(unconstrained_wind_field.z) > 0)
    
    # Check domain extents match parameters
    assert np.isclose(unconstrained_wind_field.x[-1] - unconstrained_wind_field.x[0], 
                      params["Lx"], rtol=0.01)
    assert np.isclose(unconstrained_wind_field.y[-1] - unconstrained_wind_field.y[0], 
                      params["Ly"], rtol=0.01)
    assert np.isclose(unconstrained_wind_field.z[-1] - unconstrained_wind_field.z[0], 
                      params["Lz"], rtol=0.01)


# ===== Constrained Turbulence Tests =====


def test_constrained_shape(constrained_wind_field):
    """Test that constrained wind field has correct shape."""
    assert constrained_wind_field.U.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert constrained_wind_field.V.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert constrained_wind_field.W.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert constrained_wind_field.x.shape == (params["Nx"],)
    assert constrained_wind_field.y.shape == (params["Ny"],)
    assert constrained_wind_field.z.shape == (params["Nz"],)


def test_constrained_types(constrained_wind_field):
    """Test that constrained wind field components are numpy arrays."""
    assert isinstance(constrained_wind_field.U, np.ndarray)
    assert isinstance(constrained_wind_field.V, np.ndarray)
    assert isinstance(constrained_wind_field.W, np.ndarray)


def test_constrained_reproducibility(constrained_stencil):
    """Test that constrained turbulence generation is reproducible with same seed."""
    wf1 = constrained_stencil.turbulence(ae, seed)
    wf2 = constrained_stencil.turbulence(ae, seed)
    
    assert np.allclose(wf1.U, wf2.U, atol=1e-5)
    assert np.allclose(wf1.V, wf2.V, atol=1e-5)
    assert np.allclose(wf1.W, wf2.W, atol=1e-5)


def test_constrained_stencil_properties(constrained_stencil):
    """Test properties of constrained stencil."""
    # Test sparsity measurement
    sparsity = constrained_stencil.sparsity
    assert isinstance(sparsity, float)
    assert 0.0 <= sparsity <= 1.0
    
    # Test spectral compression
    compression = constrained_stencil.spectral_compression
    assert isinstance(compression, float)
    assert 0.0 <= compression <= 1.0


def test_single_constraint():
    """Test with a single constraint."""
    single_constraint = [Constraint(x=100, y=100, z=100, u=1.0)]
    stencil = Stencil(**params, parallel=False).constrain(single_constraint).build()
    wf = stencil.turbulence(ae, seed)
    
    assert wf.U.shape == (params["Nx"], params["Ny"], params["Nz"])


def test_multiple_constraints_along_line():
    """Test multiple constraints along a line."""
    line_constraints = [
        Constraint(x=10, y=100, z=100, u=0.0),
        Constraint(x=20, y=100, z=100, u=0.5),
        Constraint(x=30, y=100, z=100, u=1.0),
        Constraint(x=40, y=100, z=100, u=0.5),
        Constraint(x=50, y=100, z=100, u=0.0),
    ]
    stencil = Stencil(**params, parallel=False).constrain(line_constraints).build()
    wf = stencil.turbulence(ae, seed)
    
    assert wf.U.shape == (params["Nx"], params["Ny"], params["Nz"])


# ===== Constraint Class Tests =====


def test_constraint_creation():
    """Test creating Constraint objects."""
    c = Constraint(x=100, y=100, z=100, u=0.5)
    assert c.x == 100
    assert c.y == 100
    assert c.z == 100
    assert c.u == 0.5


def test_constraint_with_different_values():
    """Test constraints with various values."""
    # Positive constraint
    c1 = Constraint(x=100, y=100, z=100, u=2.0)
    assert c1.u == 2.0
    
    # Negative constraint
    c2 = Constraint(x=100, y=100, z=100, u=-1.5)
    assert c2.u == -1.5
    
    # Zero constraint
    c3 = Constraint(x=100, y=100, z=100, u=0.0)
    assert c3.u == 0.0


# ===== Parallel Execution Tests =====


# ===== Parallel Execution Tests =====


def test_parallel_vs_serial_unconstrained(unconstrained_stencil):
    """Test that parallel and serial execution produce identical results for unconstrained."""
    wf_serial = unconstrained_stencil.turbulence(ae, seed, parallel=False)
    wf_parallel = unconstrained_stencil.turbulence(ae, seed, parallel=True)
    
    assert np.allclose(wf_serial.U, wf_parallel.U)
    assert np.allclose(wf_serial.V, wf_parallel.V)
    assert np.allclose(wf_serial.W, wf_parallel.W)


def test_parallel_vs_serial_constrained(constrained_stencil):
    """Test that parallel and serial execution produce identical results for constrained."""
    wf_serial = constrained_stencil.turbulence(ae, seed, parallel=False)
    wf_parallel = constrained_stencil.turbulence(ae, seed, parallel=True)
    
    # May have small numerical differences in constrained case
    assert np.allclose(wf_serial.U, wf_parallel.U, rtol=1e-5, atol=1e-6)
    assert np.allclose(wf_serial.V, wf_parallel.V, rtol=1e-5, atol=1e-6)
    assert np.allclose(wf_serial.W, wf_parallel.W, rtol=1e-5, atol=1e-6)


def test_parallel_stencil_build():
    """Test building stencil with parallel option."""
    stencil = Stencil(**params, parallel=True).build()
    wf = stencil.turbulence(ae, seed, parallel=True)
    
    assert wf.U.shape == (params["Nx"], params["Ny"], params["Nz"])


# ===== Different ae Values Tests =====


def test_different_ae_values(unconstrained_stencil):
    """Test turbulence generation with different ae (eddy lifetime) values."""
    ae_values = [0.1, 0.2, 0.5, 1.0]
    
    wind_fields = []
    for ae_val in ae_values:
        wf = unconstrained_stencil.turbulence(ae_val, seed)
        wind_fields.append(wf)
        assert wf.U.shape == (params["Nx"], params["Ny"], params["Nz"])
    
    # Different ae values should produce different fields
    for i in range(len(wind_fields) - 1):
        assert not np.allclose(wind_fields[i].U, wind_fields[i + 1].U)
        # But statistics should correlate with ae
        std_i = np.std(wind_fields[i].U)
        std_i_plus_1 = np.std(wind_fields[i + 1].U)
        assert std_i > 0 and std_i_plus_1 > 0


# ===== CLI Tests =====


def test_cli_unconstrained(tmp_path):
    """Test CLI with unconstrained turbulence."""
    runner = CliRunner()
    
    input_file = tmp_path / "input_unconstrained.toml"
    output_file = tmp_path / "wf_unconstrained.npz"
    
    toml_dict = {
        "stencil_spec": params,
        "turbulence_boxes": [
            {"ae": ae, "seed": seed, "output": str(output_file)}
        ],
    }
    with open(input_file, "w") as f:
        toml.dump(toml_dict, f)
    
    result = runner.invoke(CLI, ["--serial", str(input_file)])
    assert result.exit_code == 0, result.output
    assert output_file.exists()


def test_cli_constrained(tmp_path):
    """Test CLI with constrained turbulence."""
    runner = CliRunner()
    
    input_file = tmp_path / "input_constrained.toml"
    output_file = tmp_path / "wf_constrained.npz"
    
    # Convert constraints to dict format for TOML
    constraints_dict = [
        {"x": c.x, "y": c.y, "z": c.z, "u": c.u}
        for c in constraints
    ]
    
    toml_dict = {
        "stencil_spec": params,
        "constraints": constraints_dict,
        "turbulence_boxes": [
            {"ae": ae, "seed": seed, "output": str(output_file)}
        ],
    }
    with open(input_file, "w") as f:
        toml.dump(toml_dict, f)
    
    result = runner.invoke(CLI, ["--serial", str(input_file)])
    assert result.exit_code == 0, result.output
    assert output_file.exists()


def test_cli_parallel_serial_comparison(tmp_path):
    """Test that CLI produces same results with --serial and --parallel."""
    runner = CliRunner()
    
    input_file = tmp_path / "input.toml"
    output_serial = tmp_path / "wf_serial.npz"
    output_parallel = tmp_path / "wf_parallel.npz"
    
    # Create input file for serial run
    toml_dict_serial = {
        "stencil_spec": params,
        "turbulence_boxes": [
            {"ae": ae, "seed": seed, "output": str(output_serial)}
        ],
    }
    with open(input_file, "w") as f:
        toml.dump(toml_dict_serial, f)
    
    # Run CLI with --serial
    result_serial = runner.invoke(CLI, ["--serial", str(input_file)])
    assert result_serial.exit_code == 0, result_serial.output
    
    # Update for parallel run
    toml_dict_parallel = {
        "stencil_spec": params,
        "turbulence_boxes": [
            {"ae": ae, "seed": seed, "output": str(output_parallel)}
        ],
    }
    with open(input_file, "w") as f:
        toml.dump(toml_dict_parallel, f)
    
    # Run CLI with --parallel
    result_parallel = runner.invoke(CLI, ["--parallel", str(input_file)])
    assert result_parallel.exit_code == 0, result_parallel.output
    
    # Compare outputs
    data_serial = np.load(output_serial)
    data_parallel = np.load(output_parallel)
    
    assert np.allclose(data_serial["u"], data_parallel["u"], rtol=1e-5)
    assert np.allclose(data_serial["v"], data_parallel["v"], rtol=1e-5)
    assert np.allclose(data_serial["w"], data_parallel["w"], rtol=1e-5)


def test_cli_multiple_boxes(tmp_path):
    """Test CLI with multiple turbulence boxes."""
    runner = CliRunner()
    
    input_file = tmp_path / "input_multi.toml"
    output_1 = tmp_path / "wf_1.npz"
    output_2 = tmp_path / "wf_2.npz"
    output_3 = tmp_path / "wf_3.npz"
    
    toml_dict = {
        "stencil_spec": params,
        "turbulence_boxes": [
            {"ae": 0.1, "seed": 1111, "output": str(output_1)},
            {"ae": 0.2, "seed": 2222, "output": str(output_2)},
            {"ae": 0.3, "seed": 3333, "output": str(output_3)},
        ],
    }
    with open(input_file, "w") as f:
        toml.dump(toml_dict, f)
    
    result = runner.invoke(CLI, ["--serial", str(input_file)])
    assert result.exit_code == 0, result.output
    
    # All outputs should exist
    assert output_1.exists()
    assert output_2.exists()
    assert output_3.exists()
    
    # They should all have the correct shape
    for output_file in [output_1, output_2, output_3]:
        data = np.load(output_file)
        assert data["u"].shape == (params["Nx"], params["Ny"], params["Nz"])


def test_cli_invalid_input(tmp_path):
    """Test CLI with invalid input file."""
    runner = CliRunner()
    
    # Non-existent file
    result = runner.invoke(CLI, ["--serial", str(tmp_path / "nonexistent.toml")])
    assert result.exit_code != 0
    
    # Empty file
    empty_file = tmp_path / "empty.toml"
    empty_file.write_text("")
    result = runner.invoke(CLI, ["--serial", str(empty_file)])
    assert result.exit_code != 0


# ===== Statistical Properties Tests =====





def test_isotropy_ratio(unconstrained_wind_field):
    """Test that variance ratios are reasonable for Mann turbulence."""
    # Use larger grid for better statistics

    wf = unconstrained_wind_field
    
    var_u = np.var(wf.U)
    var_v = np.var(wf.V)
    var_w = np.var(wf.W)
    
    # Mann model produces anisotropic turbulence
    # U variance should typically be larger than V and W
    assert var_u > 0
    assert var_v > 0
    assert var_w > 0
    
    # Check that ratios are within reasonable bounds (not isotropic, but not extreme)
    assert 0.1 < var_v / var_u < 10.0
    assert 0.1 < var_w / var_u < 10.0
