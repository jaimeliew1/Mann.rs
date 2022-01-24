from . import RustMann
from pydantic import BaseModel, Extra, PositiveInt, NonNegativeFloat, PositiveFloat
import numpy as np
from pathlib import Path


class Stencil(BaseModel):
    ae: NonNegativeFloat
    L: PositiveFloat
    gamma: NonNegativeFloat
    Lx: PositiveFloat
    Ly: PositiveFloat
    Lz: PositiveFloat
    Nx: PositiveInt
    Ny: PositiveInt
    Nz: PositiveInt

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stencil = RustMann.stencilate_f64(
            self.ae,
            self.L,
            self.gamma,
            self.Lx,
            self.Ly,
            self.Lz,
            self.Nx,
            self.Ny,
            self.Nz,
        )

    class Config:
        extra = Extra.allow

    def turbulence(self, seed: int):
        U, V, W = RustMann.turbulate_f64(
            self.stencil, seed, self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz
        )

        return U, V, W


def save_box(filename, box):
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    np.array(box).astype("<f").tofile(filename)


def load_mann_binary(filename, N=(32, 32)):
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
        assert nx == int(
            nx
        ), f"Size of turbulence box ({len(data)}) does not match ny x nz ({ny*nx}), nx={nx}"
        nx = int(nx)
    else:
        nx, ny, nz = N
        assert (
            len(data) == nx * ny * nz
        ), "Size of turbulence box (%d) does not match nx x ny x nz (%d)" % (
            len(data),
            nx * ny * nz,
        )
    return data.reshape(nx, ny, nz)
