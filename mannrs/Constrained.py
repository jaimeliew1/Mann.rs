from dataclasses import dataclass
import numpy as np

from mannrs.mannrs import RustConstrainedStencil
from .Windfield import Windfield


@dataclass
class Constraint:
    """
    A velocity constraint at a specific point in space.

    Parameters
    ----------
    x : float
        X-coordinate of the constraint point (m).
    y : float
        Y-coordinate of the constraint point (m).
    z : float
        Z-coordinate of the constraint point (m).
    u : float
        Desired streamwise (x-direction) velocity at the constraint point (m/s).
    """

    x: float
    y: float
    z: float
    u: float


@dataclass
class ConstrainedStencil:
    """
    Generates a reusable Mann turbulence stencil for efficient 3D velocity field
    generation that satisfies pointwise velocity constraints.

    Parameters
    ----------
    constraints : list[Constraint]
        List of velocity constraints to enforce.
    L : float
        Turbulence length scale (m).
    gamma : float
        Shear distortion parameter (dimensionless).
    Nx : int
        Number of grid points in the x direction.
    Ny : int
        Number of grid points in the y direction.
    Nz : int
        Number of grid points in the z direction.
    Lx : float
        Domain size in x direction (m).
    Ly : float
        Domain size in y direction (m).
    Lz : float
        Domain size in z direction (m).
    aperiodic_x : bool, optional
        If True, doubles the domain in x to enforce aperiodicity.
    aperiodic_y : bool, optional
        If True, doubles the domain in y to enforce aperiodicity.
    aperiodic_z : bool, optional
        If True, doubles the domain in z to enforce aperiodicity.
    parallel : bool, optional
        Enable parallel computation.
    corr_thres : float, optional
        Correlation threshold used to truncate weak constraint influence.
    spectral_compression_target : float, optional
        Spectral impulse truncation threshold..
    sinc_thres : float, optional
        Threshold parameter used internally by the stencil algorithm.
    """

    constraints: list[Constraint]
    L: float
    gamma: float
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    aperiodic_x: bool = False
    aperiodic_y: bool = True
    aperiodic_z: bool = True
    parallel: bool = True
    corr_thres: float = 0.0001
    spectral_compression_target: float = 0.80
    sinc_thres: float = 3.0

    def __post_init__(self):
        _constraints = np.array(
            [[x.x, x.y, x.z, x.u] for x in self.constraints], dtype=np.float32
        )
        self.stencil = RustConstrainedStencil(
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
            _constraints,
            self.parallel,
            corr_thres=self.corr_thres,
            spectral_compression_target=self.spectral_compression_target,
            sinc_thres=self.sinc_thres,
        )

    def turbulence(self, ae: float, seed: int, parallel: bool = True) -> Windfield:
        """
        Generate a constrained 3D Mann turbulence field realization.

        Parameters
        ----------
        ae : float
            Turbulence intensity scaling factor (α·ε^{2/3}).
        seed : int
            Random seed for reproducibility.
        parallel : bool, optional
            Whether to use parallel computation during synthesis (default: True).

        Returns
        -------
        tuple of np.ndarray
            Velocity components (U, V, W), each of shape (Nx, Ny, Nz).
        """
        U, V, W = self.stencil.turbulate(float(ae), int(seed), parallel)
        x, y, z = self.get_axes()

        return Windfield(U, V, W, x, y, z)

    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the spatial grid axes corresponding to the generated field.

        Returns
        -------
        tuple of np.ndarray
            (x, y, z) coordinate arrays of lengths Nx, Ny, Nz respectively.
        """
        return self.stencil.get_axes()

    @property
    def sparsity(self) -> float:
        """
        Proportion of elements in the correlation matrix set to zero based on the
        corr_thres parameter.

        Returns
        -------
        float
            Fraction between 0 and 1 indicating the sparsity of the correlation
            matrix.
        """
        return self.stencil.sparsity()

    @property
    def spectral_compression(self) -> float:
        """
        Compression ratio of the constrained spectral impulse based on the
        impulse_thres parameter.

        Returns
        -------
        float
            Spectral impulse compression ratio.
        """
        return self.stencil.spectral_compression()
