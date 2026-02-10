from mannrs import Stencil
from pytest import fixture
import numpy as np
import struct
import netCDF4


# Test parameters
ae = 0.2
seed = 1234
params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 200,
    "Ly": 200,
    "Lz": 200,
    "Nx": 16,
    "Ny": 16,
    "Nz": 16,
}


@fixture
def stencil():
    """Fixture providing a built stencil for generating wind fields."""
    return Stencil(**params, parallel=False).build()


@fixture
def wind_field(stencil):
    """Fixture providing a generated wind field for testing."""
    return stencil.turbulence(ae, seed)


# ===== Wind Field Function Tests =====


def test_wind_field_shape(wind_field):
    """Test that wind field has correct shape."""
    assert wind_field.U.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wind_field.V.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wind_field.W.shape == (params["Nx"], params["Ny"], params["Nz"])
    assert wind_field.x.shape == (params["Nx"],)
    assert wind_field.y.shape == (params["Ny"],)
    assert wind_field.z.shape == (params["Nz"],)


def test_wind_field_types(wind_field):
    """Test that wind field components are numpy arrays."""
    assert isinstance(wind_field.U, np.ndarray)
    assert isinstance(wind_field.V, np.ndarray)
    assert isinstance(wind_field.W, np.ndarray)
    assert isinstance(wind_field.x, np.ndarray)
    assert isinstance(wind_field.y, np.ndarray)
    assert isinstance(wind_field.z, np.ndarray)


def test_wind_field_reproducibility(stencil):
    """Test that wind field generation is reproducible with same seed."""
    wf1 = stencil.turbulence(ae, seed)
    wf2 = stencil.turbulence(ae, seed)

    assert np.allclose(wf1.U, wf2.U)
    assert np.allclose(wf1.V, wf2.V)
    assert np.allclose(wf1.W, wf2.W)
    assert np.allclose(wf1.x, wf2.x)
    assert np.allclose(wf1.y, wf2.y)
    assert np.allclose(wf1.z, wf2.z)


def test_translate(wind_field):
    """Test the translate method."""
    y_offset = 10.0
    z_offset = 5.0

    translated = wind_field.translate(y_offset=y_offset, z_offset=z_offset)

    # Velocity components should remain unchanged
    assert np.allclose(translated.U, wind_field.U)
    assert np.allclose(translated.V, wind_field.V)
    assert np.allclose(translated.W, wind_field.W)

    # x-coordinates should remain unchanged
    assert np.allclose(translated.x, wind_field.x)

    # y and z coordinates should be offset
    assert np.allclose(translated.y, wind_field.y + y_offset)
    assert np.allclose(translated.z, wind_field.z + z_offset)


def test_velocity_offset(wind_field):
    """Test the velocity_offset method."""
    u_offset = 10.0

    offset_field = wind_field.velocity_offset(u_offset=u_offset)

    # U component should be offset
    assert np.allclose(offset_field.U, wind_field.U + u_offset)

    # V and W components should remain unchanged
    assert np.allclose(offset_field.V, wind_field.V)
    assert np.allclose(offset_field.W, wind_field.W)

    # All coordinates should remain unchanged
    assert np.allclose(offset_field.x, wind_field.x)
    assert np.allclose(offset_field.y, wind_field.y)
    assert np.allclose(offset_field.z, wind_field.z)


# ===== Output Format Tests =====


def test_write_npz(wind_field, tmp_path):
    """Test writing and reading wind field in NPZ format."""
    output_file = tmp_path / "test_field.npz"

    # Write the wind field
    wind_field.write(output_file, format="npz")

    assert output_file.exists()

    # Read back and verify
    data = np.load(output_file)

    assert np.allclose(data["u"], wind_field.U)
    assert np.allclose(data["v"], wind_field.V)
    assert np.allclose(data["w"], wind_field.W)
    assert np.allclose(data["x"], wind_field.x)
    assert np.allclose(data["y"], wind_field.y)
    assert np.allclose(data["z"], wind_field.z)


def test_write_netcdf(wind_field, tmp_path):
    """Test writing and reading wind field in NetCDF format."""
    output_file = tmp_path / "test_field.nc"

    # Write the wind field
    wind_field.write(output_file, format="netCDF")

    assert output_file.exists()

    # Read back and verify
    with netCDF4.Dataset(output_file, "r") as nc:
        assert np.allclose(nc.variables["u"][:], wind_field.U)
        assert np.allclose(nc.variables["v"][:], wind_field.V)
        assert np.allclose(nc.variables["w"][:], wind_field.W)
        assert np.allclose(nc.variables["x"][:], wind_field.x)
        assert np.allclose(nc.variables["y"][:], wind_field.y)
        assert np.allclose(nc.variables["z"][:], wind_field.z)

        # Check dimensions
        assert nc.dimensions["x"].size == params["Nx"]
        assert nc.dimensions["y"].size == params["Ny"]
        assert nc.dimensions["z"].size == params["Nz"]


def test_write_hawc2(wind_field, tmp_path):
    """Test writing wind field in HAWC2 format."""
    output_file = tmp_path / "test_field.bin"

    # Write the wind field (creates _u, _v, _w files)
    wind_field.write(output_file, format="HAWC2")

    # Check that all three component files exist
    u_file = tmp_path / "test_field_u.bin"
    v_file = tmp_path / "test_field_v.bin"
    w_file = tmp_path / "test_field_w.bin"

    assert u_file.exists()
    assert v_file.exists()
    assert w_file.exists()

    # Read back binary data and verify shape
    u_data = np.fromfile(u_file, dtype="<f")
    v_data = np.fromfile(v_file, dtype="<f")
    w_data = np.fromfile(w_file, dtype="<f")

    expected_size = params["Nx"] * params["Ny"] * params["Nz"]
    assert len(u_data) == expected_size
    assert len(v_data) == expected_size
    assert len(w_data) == expected_size

    # Verify data matches (accounting for little-endian float conversion)
    u_reshaped = u_data.reshape(wind_field.U.shape)
    v_reshaped = v_data.reshape(wind_field.V.shape)
    w_reshaped = w_data.reshape(wind_field.W.shape)

    assert np.allclose(u_reshaped, wind_field.U, rtol=1e-6)
    assert np.allclose(v_reshaped, wind_field.V, rtol=1e-6)
    assert np.allclose(w_reshaped, wind_field.W, rtol=1e-6)


def test_write_alaskawind(wind_field, tmp_path):
    """Test writing wind field in AlaskaWind format."""
    output_file = tmp_path / "test_field.wnd"

    # Write the wind field (creates _u, _v, _w files)
    wind_field.write(output_file, format="alaskaWind")

    # Check that all three component files exist
    u_file = tmp_path / "test_field_u.wnd"
    v_file = tmp_path / "test_field_v.wnd"
    w_file = tmp_path / "test_field_w.wnd"

    assert u_file.exists()
    assert v_file.exists()
    assert w_file.exists()

    # Read and verify header from one file
    with open(u_file, "rb") as f:
        # Read header: 3 integers, then 9 doubles
        Ny, Nz, Nx = struct.unpack("iii", f.read(12))
        dy, dz, dt, zmin, Uamb, zref, z0, variance, scale = struct.unpack(
            "<ddddddddd", f.read(72)
        )

        # Verify dimensions
        assert Ny == params["Ny"]
        assert Nz == params["Nz"]
        assert Nx == params["Nx"]

        # Read velocity data
        data_size = Ny * Nz * Nx
        u_data = np.fromfile(f, dtype="int16", count=data_size)

        assert len(u_data) == data_size
        assert np.allclose(u_data.sum(), wind_field.U.sum() * scale, rtol=1e-3)
