from mannrs import mannrs
import numpy as np


class Isotropic:
    def __init__(self, ae, L):
        self.ae = ae
        self.L = L

    def __repr__(self):
        return f"Tensor.Isotropic(ae={self.ae}, L={self.L})"

    def tensor(self, k):
        return mannrs.isotropic_f32(np.array(k, dtype=np.single), self.ae, self.L)

    def decomp(self, k):
        return mannrs.isotropic_sqrt_f32(np.array(k, dtype=np.single), self.ae, self.L)


class Sheared:
    def __init__(self, ae, L, gamma):
        self.ae = ae
        self.L = L
        self.gamma = gamma

    def __repr__(self):
        return f"Tensor.Sheared(ae={self.ae}, L={self.L}, gamma={self.gamma})"

    def tensor(self, k):
        return mannrs.sheared_f32(
            np.array(k, dtype=np.single), self.ae, self.L, self.gamma
        )

    def decomp(self, k):
        return mannrs.sheared_sqrt_f32(
            np.array(k, dtype=np.single), self.ae, self.L, self.gamma
        )


class ShearedSinc:
    def __init__(self, ae, L, gamma, Ly, Lz, tol, min_depth):
        self.ae = ae
        self.L = L
        self.gamma = gamma
        self.Ly = Ly
        self.Lz = Lz
        self.tol = tol
        self.min_depth = min_depth

    def __repr__(self):
        return f"Tensor.ShearedSinc(ae={self.ae}, L={self.L}, gamma={self.gamma})"

    def tensor_info(self, k):
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

    def tensor(self, k):
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

    def decomp(self, k):
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
