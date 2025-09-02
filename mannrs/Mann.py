from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from . import mannrs
from .Windfield import Windfield


def mann_spectra(
    kxs: list[float], ae: float, L: float, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 1D Mann turbulence spectra for a range of streamwise wavenumbers.

    Parameters
    ----------
    kxs : list of float
        Streamwise wavenumber values (k₁) in rad/m.
    ae : float
        Turbulence intensity scaling parameter (α·ε^{2/3}).
    L : float
        Turbulence length scale (m).
    gamma : float
        Shear distortion parameter (dimensionless).

    Returns
    -------
    tuple of np.ndarray
        Four spectral components as functions of k₁:
        - UU : Longitudinal auto-spectrum.
        - VV : Lateral auto-spectrum.
        - WW : Vertical auto-spectrum.
        - UW : Longitudinal-vertical cross-spectrum.
    """
    return mannrs.mann_spectra(np.array(kxs, dtype=np.float32), ae, L, gamma)


@dataclass
class Stencil:
    """
    Generates a reusable Mann turbulence stencil for efficient 3D velocity field generation.

    This class wraps a compiled `RustStencil` object and precomputes the structure
    required to synthesize turbulence boxes using the Mann model. The stencil
    allows rapid generation of multiple realizations with consistent spatial configuration.

    Parameters
    ----------
    L : float
        Turbulence length scale (m).
    gamma : float
        Shear distortion parameter (dimensionless).
    Lx : float
        Domain size in the streamwise (x) direction (m).
    Ly : float
        Domain size in the lateral (y) direction (m).
    Lz : float
        Domain size in the vertical (z) direction (m).
    Nx : int
        Number of grid points in the x direction.
    Ny : int
        Number of grid points in the y direction.
    Nz : int
        Number of grid points in the z direction.
    aperiodic_x : bool, optional
        If True, the turbulence box will be aperiodic in the x direction,
        achieved by doubling the stencil domain in x (default: False).
    aperiodic_y : bool, optional
        If True, the turbulence box will be aperiodic in the y direction,
        achieved by doubling the stencil domain in y (default: True).
    aperiodic_z : bool, optional
        If True, the turbulence box will be aperiodic in the z direction,
        achieved by doubling the stencil domain in z (default: True).
    parallel : bool, optional
        Enable parallel computation (default: True).
    sinc_thres : float, optional
        Threshold parameter used internally by the stencil algorithm (default: 3.0).
    """

    L: float
    gamma: float
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int
    aperiodic_x: bool = False
    aperiodic_y: bool = True
    aperiodic_z: bool = True
    parallel: bool = True
    sinc_thres: float = 3.0

    def __post_init__(self):
        self.stencil = mannrs.RustStencil(
            self.L,
            self.gamma,
            self.Lx,
            self.Ly,
            self.Lz,
            self.Nx,
            self.Ny,
            self.Nz,
            self.aperiodic_x,
            self.aperiodic_y,
            self.aperiodic_z,
            self.parallel,
            self.sinc_thres,
        )

    def turbulence(self, ae: float, seed: int, parallel=True) -> Windfield:
        """
        Generate a single realization of a 3D Mann turbulence velocity field.

        Parameters
        ----------
        ae : float
            Scaling factor related to turbulence intensity (α·ε^{2/3}).
        seed : int
            Random seed for reproducibility.
        parallel : bool, optional
            Whether to use parallel computation for this generation (default: True).

        Returns
        -------
        tuple of np.ndarray
            A tuple of 3D arrays (U, V, W), each of shape (Nx, Ny, Nz),
            representing the velocity components in the x, y, and z directions.
        """

        U, V, W = self.stencil.turbulence(ae, seed, parallel)
        x, y, z = self.get_axes()

        return Windfield(
            U[: self.Nx, : self.Ny, : self.Nz],
            V[: self.Nx, : self.Ny, : self.Nz],
            W[: self.Nx, : self.Ny, : self.Nz],
            x,
            y,
            z,
        )

    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the spatial grid axes corresponding to the generated field.

        Returns
        -------
        tuple of np.ndarray
            (x, y, z) coordinate arrays of lengths Nx, Ny, Nz respectively.
        """
        return self.stencil.get_axes()


def load_mann_binary(filename: Path, N=(32, 32)) -> ArrayLike:
    """
    Loads a mann turbulence box in HAWC2 binary format.

    Args:
        filename (str): Filename of turbulence box
        N (tuple): Number of grid points (ny, nz) or (nx, ny, nz)

    Returns:
        turbulence_box (nd_array): turbulent box data as 3D array,
    """
    data = np.fromfile(filename, np.dtype("<f"), -1)
    if len(N) == 2:
        ny, nz = N
        nx = len(data) / (ny * nz)
        assert nx == int(nx), (
            f"Size of turbulence box ({len(data)}) does not match ny x nz ({ny * nx}), nx={nx}"
        )
        nx = int(nx)
    else:
        nx, ny, nz = N
        assert len(data) == nx * ny * nz, (
            "Size of turbulence box (%d) does not match nx x ny x nz (%d)"
            % (
                len(data),
                nx * ny * nz,
            )
        )
    return data.reshape(nx, ny, nz)
