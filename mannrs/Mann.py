from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from . import mannrs


@dataclass
class Stencil:
    """
    Generate a Mann turbulence stencil.
    args:
        parallel: Use parallel operations (default: False)
    """

    L: float
    gamma: float
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int
    parallel: bool = False

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
            self.parallel,
        )

    def turbulence(
        self, ae: float, seed: int, domain="space", parallel=False
    ) -> tuple[ArrayLike, ...]:
        """
        Generate a Mann turbulence from a stencil.
        args:
            ae (float): scaling factor.
            seed (int): random seed.
            domain: return domain type. Either `space` (default) or `frequency`
            parallel: Use parallel operations (default: False)
        """
        if domain == "space":
            U, V, W = self.stencil.turbulence(
                ae,
                seed,
                parallel,
            )
        elif domain == "frequency":
            U, V, W = self.stencil.partial_turbulence(
                ae,
                seed,
                parallel,
            )

        else:
            raise ValueError

        return U, V, W


@dataclass
class ForgetfulStencil:
    """
    Generate a Mann turbulence stencil which has a low memory usage (the
    spectral tensors are not cached).
    """

    L: float
    gamma: float
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        self.stencil = mannrs.RustForgetfulStencil(
            self.L,
            self.gamma,
            self.Lx,
            self.Ly,
            self.Lz,
            self.Nx,
            self.Ny,
            self.Nz,
        )

    def turbulence(
        self, ae: float, seed: int, domain="space", parallel=False
    ) -> tuple[ArrayLike, ...]:
        """
        Generate a Mann turbulence from a stencil.
        args:
            ae (float): scaling factor.
            seed (int): random seed.
            domain: return domain type. Either `space` (default) or `frequency`
            parallel: Use parallel operations (default: False)
        """
        if domain == "space":
            U, V, W = self.stencil.turbulence(
                ae,
                seed,
                parallel,
            )
        elif domain == "frequency":
            U, V, W = self.stencil.partial_turbulence(
                ae,
                seed,
                parallel,
            )

        else:
            raise ValueError

        return U, V, W


def spectra(kxs: np.ndarray, ae: float, L: float, gamma: float) -> np.ndarray:
    Suu, Svv, Sww, Suv = mannrs.mann_spectra(np.array(kxs, dtype=np.single), ae, L, gamma)
    return Suu, Svv, Sww, Suv


def save_box(filename: Path, box: ArrayLike):
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    np.array(box).astype("<f").tofile(filename)


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
        assert (
            nx == int(nx)
        ), f"Size of turbulence box ({len(data)}) does not match ny x nz ({ny*nx}), nx={nx}"
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
