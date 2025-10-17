import numpy as np

def mann_spectra(
    kxs: np.ndarray, ae: float, L: float, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

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
        parallel: bool,
        sinc_thres: float,
    ): ...
    def constrain(
        self,
        constraints: np.ndarray,
        corr_thres: float,
        spectral_compression_target: float,
    ): ...
    def turbulence(
        self, ae: float, seed: int, parallel: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def partial_turbulence(
        self, ae: float, seed: int, parallel: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def spectral_component_grids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def correlation_grids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def spectral_impulses(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]: ...

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
        parallel: bool,
        corr_thres: float,
        spectral_compression_target: float,
        sinc_thres: float,
    ): ...
    def turbulence(
        self, ae: float, seed: int, parallel: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def sparsity(self) -> float: ...
    def spectral_compression(self) -> float: ...

def distance_matrix(x: np.ndarray) -> np.ndarray: ...
def isotropic_f32(k: np.ndarray, ae: float, L: float) -> np.ndarray: ...
def isotropic_sqrt_f32(k: np.ndarray, ae: float, L: float) -> np.ndarray: ...
def sheared_f32(k: np.ndarray, ae: float, L: float, gamma: float) -> np.ndarray: ...
def sheared_sqrt_f32(
    k: np.ndarray, ae: float, L: float, gamma: float
) -> np.ndarray: ...
def sheared_sinc_info_f32(
    k: np.ndarray,
    ae: float,
    L: float,
    gamma: float,
    Ly: float,
    Lz: float,
    tol: float,
    min_depth: float,
) -> tuple[np.ndarray, int]: ...
def sheared_sinc_f32(
    k: np.ndarray,
    ae: float,
    L: float,
    gamma: float,
    Ly: float,
    Lz: float,
    tol: float,
    min_depth: float,
) -> np.ndarray: ...
def sheared_sinc_sqrt_f32(
    k: np.ndarray,
    ae: float,
    L: float,
    gamma: float,
    Ly: float,
    Lz: float,
    tol: float,
    min_depth: float,
) -> np.ndarray: ...
