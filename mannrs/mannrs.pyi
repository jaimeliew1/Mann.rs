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
        aperiodic_x: bool,
        aperiodic_y: bool,
        aperiodic_z: bool,
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

class RustConstrainedStencil:
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
        aperiodic_x: bool,
        aperiodic_y: bool,
        aperiodic_z: bool,
        constraints: np.ndarray,
        corr_thres: float,
        sinc_thres: float,
        parallel: bool,
    ): ...
    def turbulate(
        self, ae: float, seed: int, impulse_thres: float, parallel: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def distance_matrix(x: np.ndarray) -> np.ndarray: ...

# To do: finish
