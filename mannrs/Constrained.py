from dataclasses import dataclass
from time import perf_counter
from typing import Optional


import numpy as np
from mannrs import Stencil
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator
from scipy import linalg
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
        self, seed: int, parallel: bool = False, method="rust", thres=0.0001
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

        print("interpolating contemporaneous values...")
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

        print("Solving linear system...")
        CConstUVW = linalg.solve(self.corr, (UVW_constraint - UVW_contemp))

        Nc = len(self.constraints)
        CConstU = CConstUVW[:Nc]
        CConstV = CConstUVW[Nc : 2 * Nc]
        CConstW = CConstUVW[2 * Nc :]

        Ures, Vres, Wres = np.array(U), np.array(V), np.array(W)

        print("begin superimposing constraints...")
        tstart = perf_counter()

        # spectral superposition (rust)
        if method == "rust":
            _constraints = np.array(
                [[p.x, p.y, p.z] for p in self.constraints], dtype=np.single
            )
            Uconst, Vconst, Wconst = self.stencil.stencil.constrain(
                _constraints,
                np.array(CConstU, dtype=np.single),
                np.array(CConstV, dtype=np.single),
                np.array(CConstW, dtype=np.single),
                float(thres),
                parallel,
            )
            Ures += Uconst[: self.Nx, : self.Ny, : self.Nz]
            Vres += Vconst[: self.Nx, : self.Ny, : self.Nz]
            Wres += Wconst[: self.Nx, : self.Ny, : self.Nz]

        elif method == "python":
            RUU_f, RVV_f, RWW_f, RUW_f = self.stencil.stencil.spectral_component_grids()

            kxs = np.fft.fftfreq(2 * self.Nx, self.Lx / self.Nx)
            kys = np.fft.fftfreq(2 * self.Ny, self.Ly / self.Ny)
            kzs = np.fft.rfftfreq(2 * self.Nz, self.Lz / self.Nz)

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
                U_f += 0.5 * phase * (RUU_f * CConstU[i] + RUW_f * CConstW[i])
                V_f += 0.5 * phase * (RVV_f * CConstV[i])
                W_f += 0.5 * phase * (RUW_f * CConstU[i] + RWW_f * CConstW[i])

            Ures += np.fft.irfftn(U_f)[: self.Nx, : self.Ny, : self.Nz]

        elif method == "python_reduced":
            RUU_f, RVV_f, RWW_f, RUW_f = self.stencil.stencil.spectral_component_grids()

            kxs = np.fft.fftfreq(2 * self.Nx, self.Lx / self.Nx)
            kys = np.fft.fftfreq(2 * self.Ny, self.Ly / self.Ny)
            kzs = np.fft.rfftfreq(2 * self.Nz, self.Lz / self.Nz)

            Nx_exp, Ny_exp, Nz_exp = len(kxs), len(kys), len(kzs)

            # Roll spectral components
            xroll, yroll, zroll = len(kxs) // 2, len(kys) // 2, 0
            kxs = np.roll(kxs, xroll, axis=0)
            kys = np.roll(kys, yroll, axis=0)
            kzs = np.roll(kzs, zroll, axis=0)

            RUU_f = np.roll(RUU_f, (xroll, yroll, zroll), (0, 1, 2))
            RVV_f = np.roll(RVV_f, (xroll, yroll, zroll), (0, 1, 2))
            RWW_f = np.roll(RWW_f, (xroll, yroll, zroll), (0, 1, 2))
            RUW_f = np.roll(RUW_f, (xroll, yroll, zroll), (0, 1, 2))
            # reduce spectral components TODO
            RUU_f_max, RVV_f_max, RWW_f_max = RUU_f.max(), RVV_f.max(), RWW_f.max()
            ind_x_min = np.where(RUU_f[:, yroll, zroll] >= thres * RUU_f_max)[0][0]
            ind_x_max = (
                len(kxs)
                - np.where(RUU_f[:, yroll, zroll][::-1] >= thres * RUU_f_max)[0][0]
            )

            ind_y_min = np.where(RVV_f[xroll, :, zroll] >= thres * RVV_f_max)[0][0]
            ind_y_max = (
                len(kys)
                - np.where(RVV_f[xroll, :, zroll][::-1] >= thres * RVV_f_max)[0][0]
            )

            ind_z_max = np.where(RWW_f[xroll, yroll, :] < thres * RWW_f_max)[0][0]
            print(f"ind_x_min: {ind_x_min}/{len(kxs)}")
            print(f"ind_x_max: {ind_x_max}/{len(kxs)}")
            print(f"ind_y_min: {ind_y_min}/{len(kys)}")
            print(f"ind_y_max: {ind_y_max}/{len(kys)}")
            print(f"ind_z_max: {ind_z_max}/{len(kzs)}")

            RUU_f = RUU_f[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max]
            RVV_f = RVV_f[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max]
            RWW_f = RWW_f[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max]
            RUW_f = RUW_f[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max]

            U_f, V_f, W_f = (
                np.zeros_like(RUU_f, dtype=complex),
                np.zeros_like(RUU_f, dtype=complex),
                np.zeros_like(RUU_f, dtype=complex),
            )
            kx_mesh, ky_mesh, kz_mesh = np.meshgrid(
                kxs[ind_x_min:ind_x_max],
                kys[ind_y_min:ind_y_max],
                kzs[:ind_z_max],
                indexing="ij",
            )
            for i, c in enumerate(tqdm(self.constraints)):
                phase = np.exp(
                    -2j * np.pi * (kx_mesh * c.x + ky_mesh * c.y + kz_mesh * c.z)
                )
                U_f += 0.5 * phase * (RUU_f * CConstU[i] + RUW_f * CConstW[i])
                V_f += 0.5 * phase * (RVV_f * CConstV[i])
                W_f += 0.5 * phase * (RUW_f * CConstU[i] + RWW_f * CConstW[i])


            # expand spectral components
            U_f_exp, V_f_exp, W_f_exp = (
                np.zeros((Nx_exp, Ny_exp, Nz_exp), dtype=complex),
                np.zeros((Nx_exp, Ny_exp, Nz_exp), dtype=complex),
                np.zeros((Nx_exp, Ny_exp, Nz_exp), dtype=complex),
            )
            U_f_exp[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max] = U_f
            V_f_exp[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max] = V_f
            W_f_exp[ind_x_min:ind_x_max, ind_y_min:ind_y_max, :ind_z_max] = W_f

            # Unroll spectral components TODO
            U_f_exp = np.roll(U_f_exp, (-xroll, -yroll, -zroll), (0, 1, 2))
            V_f_exp = np.roll(V_f_exp, (-xroll, -yroll, -zroll), (0, 1, 2))
            W_f_exp = np.roll(W_f_exp, (-xroll, -yroll, -zroll), (0, 1, 2))

            Ures += np.fft.irfftn(U_f_exp)[: self.Nx, : self.Ny, : self.Nz]

        elif method == "fastinterp":
            # fast interp (python)
            xmesh, ymesh, zmesh = np.meshgrid(*grid_points, indexing="ij")

            for i, c in enumerate(tqdm(self.constraints)):
                _dx = np.abs(xmesh - c.x)
                _dy = np.abs(ymesh - c.y)
                _dz = np.abs(zmesh - c.z)
                UUcorr, VVcorr, WWcorr, UWcorr = self.Rall_func(_dx, _dy, _dz)

                Ures += UUcorr * CConstU[i] + UWcorr * CConstW[i]
                Vres += VVcorr * CConstV[i]
                Wres += UWcorr * CConstU[i] + WWcorr * CConstW[i]

        else:
            raise ValueError(f"method {method} not found.")

        print(f"constraints superimposed {perf_counter() - tstart}s")
        return Ures, Vres, Wres
