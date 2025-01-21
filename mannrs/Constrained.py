from dataclasses import dataclass
from typing import Optional


import numpy as np
from mannrs import Stencil
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


@dataclass
class FastNearestNeighbor3DEquidistantInputMultiOutputInterpolator:
    outputs: list[np.array]
    dx: float
    dy: float
    dz: float

    def __post_init__(self):
        self.imax = self.outputs[0].shape[0] - 1
        self.jmax = self.outputs[0].shape[1] - 1
        self.kmax = self.outputs[0].shape[2] - 1

    def __call__(self, X: np.array, Y: np.array, Z: np.array) -> list[np.array]:
        # Calculate the indices of the nearest grid points
        i = (X // self.dx).astype(int)
        j = (Y // self.dy).astype(int)
        k = (Z // self.dz).astype(int)

        # Clip the indices to the valid range
        i = np.clip(i, 0, self.imax)
        j = np.clip(j, 0, self.jmax)
        k = np.clip(k, 0, self.kmax)
        # Extract the values at the nearest grid points
        values = [out[i, j, k] for out in self.outputs]
        return values


@dataclass
class Constraint:
    x: float
    y: float
    z: float
    u: Optional[float] = None
    v: Optional[float] = None
    w: Optional[float] = None


@dataclass
class ConstrainedStencil:
    constraints: list[Constraint]
    ae: float
    L: float
    gamma: float
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    aperiodic_x: bool = True
    aperiodic_y: bool = True
    aperiodic_z: bool = True
    parallel: bool = False

    def __post_init__(self):

        print("generating stencil...")
        self.stencil = Stencil(
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
        )

        RUU, RVV, RWW, RUW = self.stencil.stencil.correlation_grids()

        # Clip correlation data
        RUW = RUW[: self.Nx, : self.Ny, : self.Nz]
        RUU = RUU[: self.Nx, : self.Ny, : self.Nz]
        RVV = RVV[: self.Nx, : self.Ny, : self.Nz]
        RWW = RWW[: self.Nx, : self.Ny, : self.Nz]

        self.Rall_func = FastNearestNeighbor3DEquidistantInputMultiOutputInterpolator(
            [RUU, RVV, RWW, RUW],
            self.Lx / self.Nx,
            self.Ly / self.Ny,
            self.Lz / self.Nz,
        )

        Nc = len(self.constraints)
        xdist = spatial.distance_matrix(
            [[p.x] for p in self.constraints], [[p.x] for p in self.constraints]
        )
        ydist = spatial.distance_matrix(
            [[p.y] for p in self.constraints], [[p.y] for p in self.constraints]
        )
        zdist = spatial.distance_matrix(
            [[p.z] for p in self.constraints], [[p.z] for p in self.constraints]
        )

        UUcorr, VVcorr, WWcorr, UWcorr = self.Rall_func(xdist, ydist, zdist)

        _zeros = np.zeros_like(UUcorr)

        corr = np.block(
            [
                [UUcorr, _zeros, UWcorr],
                [_zeros, VVcorr, _zeros],
                [UWcorr, _zeros, WWcorr],
            ]
        )

        self.corr = corr

    def turbulence(
        self, seed: int, parallel: bool = False
    ) -> tuple[np.array, np.array, np.array]:
        U, V, W = self.stencil.turbulence(self.ae, seed, parallel=parallel)

        grid_points = (
            np.linspace(0, self.Lx, self.Nx),
            np.linspace(0, self.Ly, self.Ny),
            np.linspace(0, self.Lz, self.Nz),
        )

        U_interp = RegularGridInterpolator(grid_points, U)
        V_interp = RegularGridInterpolator(grid_points, V)
        W_interp = RegularGridInterpolator(grid_points, W)

        U_contemp = U_interp([(p.x, p.y, p.z) for p in self.constraints])
        V_contemp = V_interp([(p.x, p.y, p.z) for p in self.constraints])
        W_contemp = W_interp([(p.x, p.y, p.z) for p in self.constraints])

        UVW_contemp = np.concatenate([U_contemp, V_contemp, W_contemp])

        UVW_constraint = np.concatenate(
            [
                [p.u for p in self.constraints],
                [p.v for p in self.constraints],
                [p.w for p in self.constraints],
            ]
        )
        UVW_constraint = np.array([x or y for x, y in zip(UVW_constraint, UVW_contemp)])

        CConstUVW = np.linalg.solve(self.corr, (UVW_constraint - UVW_contemp))

        Nc = len(self.constraints)
        CConstU = CConstUVW[:Nc]
        CConstV = CConstUVW[Nc : 2 * Nc]
        CConstW = CConstUVW[2 * Nc :]

        Ures, Vres, Wres = np.array(U), np.array(V), np.array(W)

        xmesh, ymesh, zmesh = np.meshgrid(*grid_points, indexing="ij")

        RUU_f, RVV_f, RWW_f, RUW_f = self.stencil.stencil.spectral_component_grids()


        kxs = np.fft.fftfreq(2*self.Nx, self.Lx / self.Nx)
        kys = np.fft.fftfreq(2*self.Ny, self.Ly / self.Ny)
        kzs = np.fft.rfftfreq( 2*self.Nz, self.Lz / self.Nz)

        U_f, V_f, W_f = (
            np.zeros_like(RUU_f, dtype=complex),
            np.zeros_like(RUU_f, dtype=complex),
            np.zeros_like(RUU_f, dtype=complex),
        )
        kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kxs, kys, kzs, indexing="ij")
        for i, c in enumerate(tqdm(self.constraints)):
            phase = np.exp(
                -2j * np.pi * (kx_mesh * c.x + ky_mesh * c.y + kz_mesh * c.z)
            )
            U_f += phase * (RUU_f * CConstU[i] + RUW_f * CConstW[i])
            V_f += phase * (RVV_f * CConstV[i])
            W_f += phase * (RUW_f * CConstU[i] + RWW_f * CConstW[i])

        Ures += np.fft.irfftn(U_f)[: self.Nx, : self.Ny, : self.Nz]

        # for i, c in enumerate(tqdm(self.constraints)):
        #     _dx = np.abs(xmesh - c.x)
        #     _dy = np.abs(ymesh - c.y)
        #     _dz = np.abs(zmesh - c.z)
        #     UUcorr, VVcorr, WWcorr, UWcorr = self.Rall_func(_dx, _dy, _dz)

        #     Ures += UUcorr * CConstU[i] + UWcorr * CConstW[i]
        #     Vres += VVcorr * CConstV[i]
        #     Wres += UWcorr * CConstU[i] + WWcorr * CConstW[i]

        return Ures, Vres, Wres
