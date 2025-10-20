from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from . import mannrs


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
class Isotropic:
    """
    Isotropic Mann spectral tensor.

    Represents a homogeneous and isotropic turbulence model using the
    Mann formulation with a given spectral scaling factor and length scale.
    """

    ae: float
    """Spectral scaling factor (αε^{2/3}), controls turbulence intensity."""
    L: float
    """Length scale (m), characterizes the size of energy-containing eddies."""

    def tensor(self, k: tuple[float, float, float]) -> ArrayLike:
        """
        Compute the isotropic spectral tensor Φ_ij(k).

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        ArrayLike
            3x3 spectral tensor for the given wavenumber.
        """
        return mannrs.isotropic_f32(np.array(k, dtype=np.single), self.ae, self.L)

    def decomp(self, k: tuple[float, float, float]) -> ArrayLike:
        """
        Compute a square-root decomposition of the isotropic spectral tensor.

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        ArrayLike
            Square-root factor of the 3x3 spectral tensor.
        """
        return mannrs.isotropic_sqrt_f32(np.array(k, dtype=np.single), self.ae, self.L)


@dataclass
class Sheared:
    """
    Sheared Mann spectral tensor.

    Extends the isotropic tensor with an anisotropy parameter `gamma` that
    elongates turbulence structures along the streamwise direction.
    """

    ae: float
    """Spectral scaling factor (αε^{2/3}), controls turbulence intensity."""
    L: float
    """Length scale (m), characterizes the size of energy-containing eddies."""
    gamma: float
    """Anisotropy parameter, controls the shear-induced elongation."""

    def tensor(self, k: tuple[float, float, float]) -> ArrayLike:
        """
        Compute the sheared spectral tensor Φ_ij(k).

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        ArrayLike
            3x3 spectral tensor for the given wavenumber.
        """
        return mannrs.sheared_f32(
            np.array(k, dtype=np.single), self.ae, self.L, self.gamma
        )

    def decomp(self, k: tuple[float, float, float]) -> ArrayLike:
        """
        Compute a square-root decomposition of the sheared spectral tensor.

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        ArrayLike
            Square-root factor of the 3x3 spectral tensor.
        """
        return mannrs.sheared_sqrt_f32(
            np.array(k, dtype=np.single), self.ae, self.L, self.gamma
        )


@dataclass
class ShearedSinc:
    """
    Sheared Mann spectral tensor with sinc correction.

    Applies a sinc filter to reduce aliasing in truncated domains.
    Suitable for simulations where periodicity or limited resolution
    affects the low-frequency content of the turbulence field.
    """

    ae: float
    """Spectral scaling factor (αε^{2/3}), controls turbulence intensity."""
    L: float
    """Length scale (m), characterizes the size of energy-containing eddies."""
    gamma: float
    """Anisotropy parameter, controls the shear-induced elongation."""
    Ly: float
    """Domain length in the lateral (y) direction."""
    Lz: float
    """Domain length in the vertical (z) direction."""
    tol: float
    """Tolerance for the adaptive quadrature used in evaluating the sinc correction. 
    Smaller values increase accuracy at the cost of performance."""
    min_depth: float
    """Minimum recursion depth for adaptive quadrature integration."""

    def tensor_info(self, k: tuple[float, float, float]) -> tuple[ArrayLike, int]:
        """
        Compute the sheared-sinc spectral tensor and metadata.

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        tensor : ArrayLike
            3x3 spectral tensor for the given wavenumber.
        depth : int
            Number of recursion levels used in the sinc evaluation.
        """
        return mannrs.sheared_sinc_info_f32(
            np.array(k, dtype=np.single),
            self.ae,
            self.L,
            self.gamma,
            self.Ly,
            self.Lz,
            self.tol,
            self.min_depth,
        )

    def tensor(self, k: tuple[float, float, float]) -> ArrayLike:
        """
        Compute the sheared-sinc spectral tensor Φ_ij(k).

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        ArrayLike
            3x3 spectral tensor for the given wavenumber.
        """
        return mannrs.sheared_sinc_f32(
            np.array(k, dtype=np.single),
            self.ae,
            self.L,
            self.gamma,
            self.Ly,
            self.Lz,
            self.tol,
            self.min_depth,
        )

    def decomp(self, k: tuple[float, float, float]) -> ArrayLike:
        """
        Compute a square-root decomposition of the sheared-sinc spectral tensor.

        Parameters
        ----------
        k : tuple of float
            Wavenumber vector (kx, ky, kz).

        Returns
        -------
        ArrayLike
            Square-root factor of the 3x3 spectral tensor.
        """
        return mannrs.sheared_sinc_sqrt_f32(
            np.array(k, dtype=np.single),
            self.ae,
            self.L,
            self.gamma,
            self.Ly,
            self.Lz,
            self.tol,
            self.min_depth,
        )
