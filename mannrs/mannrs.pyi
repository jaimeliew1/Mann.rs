import numpy as np

class RustStencil:
    def __init__(
        self,
        L: float,
        gamma: float,
        Lx: float,
        Ly: float,
        Lz: float,
        Nx: int,
        Ny: int,
        Nz: int,
        sinc_thres: float,
    ): ...
    def turbulence(
        self, ae: float, seed: int, parallel=bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def partial_turbulence(
        self, ae: float, seed: int, parallel=bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def spectral_component_grids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def correlation_grids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def constrain(
        self,
        constraints: np.ndarray,
        CConstU: np.ndarray,
        CConstV: np.ndarray,
        CConstW: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...



# To do: finish