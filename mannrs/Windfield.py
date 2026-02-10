from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import netCDF4
import struct


@dataclass
class Windfield:
    """
    A container for generated wind field data.
    Examples
    --------
    Save to file:
    >>> wf.write("turbulence_field.nc", format="netCDF")

    Access individual components:

    >>> u_velocity = wf.U  # Streamwise component
    >>> v_velocity = wf.V  # Lateral component
    >>> w_velocity = wf.W  # Vertical component
    >>> x_coords = wf.x    # X-coordinates
    """

    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        # Check that U, V, W are 3D and have the same shape
        if not (self.U.ndim == self.V.ndim == self.W.ndim == 3):
            raise ValueError("U, V, and W must be 3D arrays.")
        if not (self.U.shape == self.V.shape == self.W.shape):
            raise ValueError("U, V, and W must have the same shape.")

        # Check that x, y, z are 1D arrays
        if not (self.x.ndim == self.y.ndim == self.z.ndim == 1):
            raise ValueError("x, y, and z must be 1D arrays.")

        # Check that the lengths of x, y, z match the dimensions of U
        if not (
            self.U.shape[0] == len(self.x)
            and self.U.shape[1] == len(self.y)
            and self.U.shape[2] == len(self.z)
        ):
            raise ValueError("Dimensions of U, V, W must match lengths of x, y, z.")

    def __repr__(self):
        return f"Windfield(Nx={len(self.x)}, Ny={len(self.y)}, Nz={len(self.z)})"

    def translate(self, y_offset: float = 0.0, z_offset: float = 0.0) -> Windfield:
        """
        Translate the wind field in the y and z directions.

        Parameters
        ----------
        y_offset : float, optional
            Distance to translate the field in the y-direction (default is 0.0).
        z_offset : float, optional
            Distance to translate the field in the z-direction (default is 0.0).

        Returns
        -------
        Windfield
            A new Windfield instance with translated coordinates.
        """
        return Windfield(
            U=self.U,
            V=self.V,
            W=self.W,
            x=self.x,
            y=self.y + y_offset,
            z=self.z + z_offset,
        )

    def velocity_offset(self, u_offset: float = 0.0) -> Windfield:
        """
        Apply a constant offset to the u-component of the velocity field.

        Parameters
        ----------
        u_offset : float, optional
            Value to add to the u-component of the velocity field (default is 0.0).

        Returns
        -------
        Windfield
            A new Windfield instance with the u-component offset applied.
        """
        return Windfield(
            U=self.U + u_offset,
            V=self.V,
            W=self.W,
            x=self.x,
            y=self.y,
            z=self.z,
        )

    def write(
        self,
        filename: Union[str, Path],
        format: Literal["npz", "netCDF", "HAWC2", "alaskaWind"] = "npz",
    ) -> None:
        """
        Write the turbulence field to disk in one of several supported formats.

        Parameters
        ----------
        filename : Union[str, Path]
            Target file path. For HAWC2 output, this stem will be used to
            generate three files (``*_u``, ``*_v``, ``*_w``).
        format : {"npz", "netCDF", "HAWC2", "alaskaWind"}, default="npz"
            Output format:

            - "npz"    : NumPy archive.

            - "netCDF" : NetCDF format.

            - "HAWC2"  : Three component files suitable for HAWC2.

            - "alaskaWind" : Binary format for Alaska wind field files.


        Notes
        -----
        - In "HAWC2" mode, three separate files are created for the velocity
          components (u, v, w) with suffixes appended to the given filename stem.
        """

        # Create directory if it does not exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        if format == "npz":
            self.to_npz(filename)
        elif format == "netCDF":
            self.to_netCDF(filename, Uamb=0.0)
        elif format == "HAWC2":
            filename = Path(filename)
            _stem = filename.stem
            self.to_HAWC2(
                filename.with_stem(_stem + "_u"),
                filename.with_stem(_stem + "_v"),
                filename.with_stem(_stem + "_w"),
            )
        elif format == "alaskaWind":
            filename = Path(filename)
            _stem = filename.stem
            self.to_wnd(
                filename.with_stem(_stem + "_u"),
                filename.with_stem(_stem + "_v"),
                filename.with_stem(_stem + "_w"),
                Uamb=0.0,
            )

    def to_HAWC2(
        self,
        fn_u: Union[str, Path],
        fn_v: Union[str, Path],
        fn_w: Union[str, Path],
    ) -> None:
        """
        Export the wind field to HAWC2 binary format.

        This method writes the 3D velocity components (U, V, W) to separate binary files
        in a format compatible with HAWC2's user-defined wind inflow.

        Parameters
        ----------
        fn_u : str or Path
            Path to the output file for the U-component of velocity.
        fn_v : str or Path
            Path to the output file for the V-component of velocity.
        fn_w : str or Path
            Path to the output file for the W-component of velocity.
        U_offset : float, optional
            Offset added to the U-component of velocity before writing (default is 0.0).
        """
        np.array(self.U).astype("<f").tofile(fn_u)
        np.array(self.V).astype("<f").tofile(fn_v)
        np.array(self.W).astype("<f").tofile(fn_w)

    def to_npz(self, fn: Union[str, Path]) -> None:
        np.savez(
            fn,
            allow_pickle=False,
            u=self.U,
            v=self.V,
            w=self.W,
            x=self.x,
            y=self.y,
            z=self.z,
        )

    def to_netCDF(self, fn: Union[str, Path], Uamb: float) -> None:
        """
        Export the wind field to a NetCDF file.

        This method writes the 3D velocity components (U, V, W) and spatial coordinates
        (x, y, z) to a NetCDF file. The time dimension is derived from the x-coordinate
        and the ambient wind speed `Uamb`.

        Parameters
        ----------
        fn : str or Path
            Path to the output NetCDF file.
        Uamb : float
            Ambient wind speed used to convert x-coordinates to time.

        - The NetCDF file will contain dimensions: time, x, y, z and variables: u, v, w, x, y, z, time.
        """

        ncfile = netCDF4.Dataset(fn, "w", format="NETCDF4")

        t = self.x * Uamb

        # Define dimensions
        ncfile.createDimension("time", len(t))
        ncfile.createDimension("x", len(self.x))
        ncfile.createDimension("y", len(self.y))
        ncfile.createDimension("z", len(self.z))

        # Define variables
        nc_t = ncfile.createVariable("time", np.float64, ("time",))
        nc_x = ncfile.createVariable("x", np.float64, ("x",))
        nc_y = ncfile.createVariable("y", np.float64, ("y",))
        nc_z = ncfile.createVariable("z", np.float64, ("z",))

        nc_u = ncfile.createVariable("u", np.float64, ("x", "y", "z"))
        nc_v = ncfile.createVariable("v", np.float64, ("x", "y", "z"))
        nc_w = ncfile.createVariable("w", np.float64, ("x", "y", "z"))

        nc_t[:] = t
        nc_x[:] = self.x
        nc_y[:] = self.y
        nc_z[:] = self.z

        nc_u[:, :, :] = self.U
        nc_v[:, :, :] = self.V
        nc_w[:, :, :] = self.W

        ncfile.close()

    def to_wnd(
        self,
        fn_u: Union[str, Path],
        fn_v: Union[str, Path],
        fn_w: Union[str, Path],
        Uamb: float,
    ) -> None:
        to_wnd_single(self.U, self.x, self.y, self.z, fn_u, Uamb)
        to_wnd_single(self.V, self.x, self.y, self.z, fn_v, Uamb)
        to_wnd_single(self.W, self.x, self.y, self.z, fn_w, Uamb)


def to_wnd_single(
    U: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, fn: Path, Uamb: float
) -> None:
    """
    Write a single velocity component to Alaska wind field binary format.

    This helper function exports one velocity component (U, V, or W) to a binary
    file compatible with Alaska/Wind software. The data is stored as 16-bit integers
    with an appropriate scaling factor.

    Parameters
    ----------
    U : np.ndarray
        3D array of velocity values with shape (Nx, Ny, Nz).
    x : np.ndarray
        1D array of x-coordinates (streamwise direction).
    y : np.ndarray
        1D array of y-coordinates (lateral direction).
    z : np.ndarray
        1D array of z-coordinates (vertical direction).
    fn : Path
        Path to the output binary file.
    Uamb : float
        Ambient wind speed used to compute time step from spatial coordinates.

    Notes
    -----
    The binary file format includes a header with grid dimensions and parameters,
    followed by the velocity data in (y, z, time) order as 16-bit integers.
    """

    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    dt = Uamb * (x[-1] - x[0]) / (Nx - 1)
    dy = (y[-1] - y[0]) / (Ny - 1)
    dz = (z[-1] - z[0]) / (Nz - 1)
    
    # Reference height for wind shear (not applicable here)
    zref = 0.0
    
    # Surface roughness length (not applicable here)
    z0 = 0.0
    
    # Scale factor for 16-bit integer representation
    scale = 30000 / np.max(np.abs(U))

    # Transpose to (y, z, time) order expected by AlaskaWind
    U_transposed = np.transpose(U, axes=(1, 2, 0))

    # Scale to integer representation and flatten
    U_transformed = (U_transposed.reshape(-1) * scale).astype("int16")

    with open(fn, "wb") as fid:
        fid.write(struct.pack("iii", Ny, Nz, Nx))
        fid.write(
            struct.pack(
                "<ddddddddd",
                dy,
                dz,
                dt,
                z.min(),
                Uamb,
                zref,
                z0,
                U.var(),
                scale,
            )
        )

        U_transformed.tofile(fid)
