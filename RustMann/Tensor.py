from RustMann import RustMann
import numpy as np


class Isotropic:
    def __init__(self, ae, L):
        self.ae = ae
        self.L = L

    def __repr__(self):
        return f"Tensor.Isotropic(ae={self.ae}, L={self.L})"

    def tensor(self, k):
        return RustMann.isotropic_f64(np.array(k, dtype=float), self.ae, self.L)

    def decomp(self, k):
        return RustMann.isotropic_sqrt_f64(np.array(k, dtype=float), self.ae, self.L)


class Sheared:
    def __init__(self, ae, L, gamma):
        self.ae = ae
        self.L = L
        self.gamma = gamma

    def __repr__(self):
        return f"Tensor.Sheared(ae={self.ae}, L={self.L}, gamma={self.gamma})"

    def tensor(self, k):
        return RustMann.sheared_f64(
            np.array(k, dtype=float), self.ae, self.L, self.gamma
        )

    def decomp(self, k):
        return RustMann.sheared_sqrt_f64(
            np.array(k, dtype=float), self.ae, self.L, self.gamma
        )
