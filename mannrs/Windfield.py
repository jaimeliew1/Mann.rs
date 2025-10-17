from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import netCDF4


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

    def write(
        self,
        filename: Union[str, Path],
        format: Literal["npz", "netCDF", "HAWC2"] = "npz",
        u_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
    ) -> None:
        """
        Write the turbulence field to disk in one of several supported formats.

        Parameters
        ----------
        filename : Union[str, Path]
            Target file path. For HAWC2 output, this stem will be used to
            generate three files (``*_u``, ``*_v``, ``*_w``).
        format : {"npz", "netCDF", "HAWC2"}, default="npz"
            Output format:

            - "npz"    : NumPy archive.

            - "netCDF" : NetCDF format.

            - "HAWC2"  : Three component files suitable for HAWC2.
        u_offset : float, default=0.0
            Constant offset to add to the u-component of the velocity field.
        y_offset : float, default=0.0
            Spatial offset applied to the y-axis.
        z_offset : float, default=0.0
            Spatial offset applied to the z-axis.

        Notes
        -----
        - In "HAWC2" mode, three separate files are created for the velocity
          components (u, v, w) with suffixes appended to the given filename stem.
        """
        if format == "npz":
            self.to_npz(
                filename, U_offset=u_offset, y_offset=y_offset, z_offset=z_offset
            )
        elif format == "netCDF":
            self.to_netCDF(
                filename,
                Uamb=0.0,
                U_offset=u_offset,
                y_offset=y_offset,
                z_offset=z_offset,
            )
        elif format == "HAWC2":
            filename = Path(filename)
            _stem = filename.stem
            self.to_HAWC2(
                filename.with_stem(_stem + "_u"),
                filename.with_stem(_stem + "_v"),
                filename.with_stem(_stem + "_w"),
                U_offset=u_offset,
            )

    def to_HAWC2(
        self,
        fn_u: Union[str, Path],
        fn_v: Union[str, Path],
        fn_w: Union[str, Path],
        U_offset: float = 0.0,
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
        np.array(self.U + U_offset).astype("<f").tofile(fn_u)
        np.array(self.V).astype("<f").tofile(fn_v)
        np.array(self.W).astype("<f").tofile(fn_w)

    def to_npz(
        self,
        fn: Union[str, Path],
        U_offset: float = 0.0,
        z_offset: float = 0.0,
        y_offset: float = 0.0,
    ) -> None:
        np.savez(
            fn,
            allow_pickle=False,
            u=self.U + U_offset,
            v=self.V,
            w=self.W,
            x=self.x,
            y=self.y + y_offset,
            z=self.z + z_offset,
        )

    def to_netCDF(
        self,
        fn: Union[str, Path],
        Uamb: float,
        U_offset: float = 0.0,
        z_offset: float = 0.0,
        y_offset: float = 0.0,
    ) -> None:
        """
        Export the wind field to a NetCDF file.

        This method writes the 3D velocity components (U, V, W) and spatial coordinates
        (x, y, z) to a NetCDF file. The time dimension is derived from the x-coordinate
        and the ambient wind speed `Uamb`. Optional offsets can be applied to the velocity
        and spatial coordinates.

        Parameters
        ----------
        fn : str or Path
            Path to the output NetCDF file.
        Uamb : float
            Ambient wind speed used to convert x-coordinates to time.
        U_offset : float, optional
            Offset added to the U-component of velocity.
        z_offset : float, optional
            Offset added to the z-coordinate.
        y_offset : float, optional
            Offset added to the y-coordinate.

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
        nc_y[:] = self.y + y_offset
        nc_z[:] = self.z + z_offset

        nc_u[:, :, :] = self.U + U_offset
        nc_v[:, :, :] = self.V
        nc_w[:, :, :] = self.W

        ncfile.close()
