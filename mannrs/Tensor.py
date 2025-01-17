from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from . import mannrs


@dataclass
class Isotropic:
    ae: float
    L: float


    def tensor(self, k: tuple[float, float, float]) -> ArrayLike:
        return mannrs.isotropic_f32(np.array(k, dtype=np.single), self.ae, self.L)

    def decomp(self, k: tuple[float, float, float]) -> ArrayLike:
        return mannrs.isotropic_sqrt_f32(np.array(k, dtype=np.single), self.ae, self.L)

@dataclass
class Sheared:
    ae: float
    L: float
    gamma: float


    def tensor(self, k: tuple[float, float, float]) -> ArrayLike:
        return mannrs.sheared_f32(
            np.array(k, dtype=np.single), self.ae, self.L, self.gamma
        )

    def decomp(self, k: tuple[float, float, float]) -> ArrayLike:
        return mannrs.sheared_sqrt_f32(
            np.array(k, dtype=np.single), self.ae, self.L, self.gamma
        )


@dataclass
class ShearedSinc:
    ae: float
    L: float
    gamma: float
    Ly: float
    Lz: float
    tol: float
    min_depth: float
    

    def tensor_info(self, k: tuple[float, float, float]) -> tuple[ArrayLike, int]:
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
