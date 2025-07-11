from dataclasses import dataclass
import numpy as np

from mannrs.mannrs import RustConstrainedStencil


@dataclass
class Constraint:
    x: float
    y: float
    z: float
    u: float


@dataclass
class ConstrainedStencil:
    constraints: list[Constraint]
    L: float
    gamma: float
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    aperiodic_x: bool = False
    aperiodic_y: bool = True
    aperiodic_z: bool = True
    parallel: bool = True
    corr_thres: float = 0.0001
    impulse_thres: float = 0.0005
    sinc_thres: float = 3.0

    def __post_init__(self):
        print("generating stencil...")
        _constraints = np.array(
            [[x.x, x.y, x.z, x.u] for x in self.constraints], dtype=np.float32
        )
        self.stencil = RustConstrainedStencil(
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
            _constraints,
            self.parallel,
            corr_thres=self.corr_thres,
            impulse_thres=self.impulse_thres,
            sinc_thres=self.sinc_thres,
        )

    def turbulence(
        self, ae: float, seed: int, parallel: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.stencil.turbulate(float(ae), int(seed), parallel)

    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.stencil.get_axes()

    @property
    def sparsity(self) -> float:
        return self.stencil.sparsity()

    @property
    def spectral_compression(self) -> float:
        return self.stencil.spectral_compression()
