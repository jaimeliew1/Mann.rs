from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal, Optional, Union

import numpy as np
import toml
from pydantic import BaseModel, Field, ValidationError

from .mannrs import RustConstrainedStencil, RustStencil
from .Windfield import Windfield


class Stencil(BaseModel, extra="allow"):
    """
    Main entry point for defining a turbulence stencil, constraints, and turbulence field generation specs.
    """
    stencil_spec: StencilSpec
    """Parameters for the turbulence stencil."""
    constraint_spec: Optional[ConstraintSpec] = None
    """Optional velocity constraints applied to the stencil."""
    turbulence_boxes: Optional[list[TurbulenceSpec]] = None
    """Optional list of TurbulenceSpec objects for wind field generation."""

    def __init__(self, **kwargs) -> None:
        try:
            super().__init__(stencil_spec=StencilSpec(**kwargs))
        except ValidationError:
            super().__init__(**kwargs)

    def constrain(
        self,
        constraints: list[Constraint],
        spectral_compression_target: float = 0.8,
        corr_thres: float = 0.0001,
    ) -> Stencil:
        """
        Apply constraints to the stencil.
        Returns a ConstrainedStencil object.
        """
        return Stencil(
            stencil_spec=self.stencil_spec,
            constraint_spec=ConstraintSpec(
                constraints=constraints,
                spectral_compression_target=spectral_compression_target,
                corr_thres=corr_thres,
            ),
        )

    @classmethod
    def from_toml(cls, filepath: Union[str, Path]) -> Stencil:
        """
        Load stencil parameters (and optional constraints) from a TOML file.
        """
        with open(filepath, "r") as f:
            data = toml.load(f)
        sim = Stencil(**data)
        return sim

    @property
    def constrained(self) -> bool:
        return self.constraint_spec is not None

    def build(self, parallel: bool = True) -> StencilInstance:
        """
        Materialize the Mann turbulence stencil in memory.
        """
        benchmark = {}
        tstart = perf_counter()
        if self.constraint_spec is not None:
            _constraints = np.array(
                [[x.x, x.y, x.z, x.u] for x in self.constraint_spec.constraints],
                dtype=np.float32,
            )
            stencil = RustConstrainedStencil(
                self.stencil_spec.L,
                self.stencil_spec.gamma,
                self.stencil_spec.Lx,
                self.stencil_spec.Ly,
                self.stencil_spec.Lz,
                self.stencil_spec.Nx,
                self.stencil_spec.Ny,
                self.stencil_spec.Nz,
                self.stencil_spec.aperiodic_x,
                self.stencil_spec.aperiodic_y,
                self.stencil_spec.aperiodic_z,
                _constraints,
                parallel,
                corr_thres=self.constraint_spec.corr_thres,
                spectral_compression_target=self.constraint_spec.spectral_compression_target,
                sinc_thres=self.stencil_spec.sinc_thres,
            )
            benchmark["sparsity"] = stencil.sparsity()
            benchmark["spectral_compression"] = stencil.spectral_compression()
        else:
            stencil = RustStencil(
                self.stencil_spec.L,
                self.stencil_spec.gamma,
                self.stencil_spec.Lx,
                self.stencil_spec.Ly,
                self.stencil_spec.Lz,
                self.stencil_spec.Nx,
                self.stencil_spec.Ny,
                self.stencil_spec.Nz,
                self.stencil_spec.aperiodic_x,
                self.stencil_spec.aperiodic_y,
                self.stencil_spec.aperiodic_z,
                parallel,
                self.stencil_spec.sinc_thres,
            )

        benchmark["stencil_time"] = perf_counter() - tstart

        return StencilInstance(stencil, benchmark, self)

    def turbulence(self, *_, **__):
        raise RuntimeError(
            "Cannot call `.turbulence()` on a Stencil object. "
            "Call `.build()` first to get a StencilInstance, then call `.turbulence()` on that."
        )


class StencilInstance:
    """
    Built stencil, ready to generate turbulence fields.
    """

    def __init__(
        self,
        stencil: Union[RustStencil, RustConstrainedStencil],
        benchmark: dict,
        params: Stencil,
    ):
        self.stencil: Union[RustStencil, RustConstrainedStencil] = stencil
        self.params: Stencil = params
        self.benchmark: dict = benchmark
        self.sparsity: Union[float, None] = benchmark.get("sparsity")
        self.spectral_compression: Union[float, None] = benchmark.get("spectral_compression")
        self.stencil_time: Union[float, None] = benchmark.get("stencil_time")

    def get_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the spatial grid axes corresponding to the generated field.

        Returns
        -------
        tuple of np.ndarray
            (x, y, z) coordinate arrays of lengths Nx, Ny, Nz respectively.
        """
        return self.stencil.get_axes()

    def turbulence(self, ae: float, seed: int, parallel: bool = True) -> Windfield:
        """
        Generate a constrained 3D Mann turbulence field realization.

        Parameters
        ----------
        ae : float
            Turbulence intensity scaling factor (α·ε^{2/3}).
        seed : int
            Random seed for reproducibility.
        parallel : bool, optional
            Whether to use parallel computation during synthesis (default: True).

        Returns
        -------
        Windfield
            Resultant turbulent velocity field.
        """
        U, V, W = self.stencil.turbulence(float(ae), int(seed), parallel)
        x, y, z = self.get_axes()

        return Windfield(U, V, W, x, y, z)


class StencilSpec(BaseModel, extra="allow"):
    """
    Base stencil template. Can be constrained via `constrain`.
    """

    L: float
    """Length scale (m), characterizes the size of energy-containing eddies"""
    gamma: float
    """Anisotropy parameter [-]"""
    Lx: float
    """Domain length in x-direction (m)"""
    Ly: float
    """Domain length in y-direction (m)"""
    Lz: float
    """Domain length in z-direction (m)"""
    Nx: int
    """Number of grid points in x-direction"""
    Ny: int
    """Number of grid points in y-direction"""
    Nz: int
    """Number of grid points in z-direction"""
    aperiodic_x: bool = False
    """sets aperiodicity in the x-direction. Turning off aperiodicity (false) can reduce computational cost by approximately half."""
    aperiodic_y: bool = True
    """sets aperiodicity in the y-direction. Turning off aperiodicity (false) can reduce computational cost by approximately half."""
    aperiodic_z: bool = True
    """sets aperiodicity in the z-direction. Turning off aperiodicity (false) can reduce computational cost by approximately half."""
    sinc_thres: float = 3.0
    """Threshold for applying the Mann sinc correction to low-frequency modes."""


class ConstraintSpec(BaseModel, extra="allow"):
    """
    Specification for velocity constraints and related parameters.
    """
    constraints: list[Constraint] = Field(..., repr=False)
    """List of velocity constraints at certain positions in 3D space."""
    spectral_compression_target: float = 0.8
    """Desired compression ratio for the constraint impulse response."""
    corr_thres: float = 0.0001
    """Threshold for sparsifying the constraint correlation matrix."""


class Constraint(BaseModel, extra="allow"):
    """
    A velocity constraint at a specific point in space.
    """

    x: float
    """X-coordinate of the constraint point (m)."""
    y: float
    """Y-coordinate of the constraint point (m)."""
    z: float
    """Z-coordinate of the constraint point (m)."""
    u: float
    """Desired streamwise (x-direction) velocity at the constraint point (m/s)."""

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(**kwargs)
        except ValidationError:
            if len(args) == 4:
                super().__init__(x=args[0], y=args[1], z=args[2], u=args[3])
            else:
                raise


class TurbulenceSpec(BaseModel, extra="allow"):
    """
    Parameters for generating and saving a turbulence wind field realization.
    """
    ae: float
    """Turbulence intensity scaling factor."""
    seed: int
    """Random seed for reproducibility."""
    output: Path
    """Output file path for saving the wind field."""
    format: Literal["npz", "netCDF", "HAWC2"] = "npz"
    """Output file format."""
    u_offset: float = 0.0
    """Velocity offset added to the u velocity component."""
    y_offset: float = 0.0
    """Spatial offset added to the y axis (e.g. to center the box at zero.)"""
    z_offset: float = 0.0
    """Spatial offset added to the z axis."""

    def generate_and_save(
        self, stencil: StencilInstance, parallel: bool = True
    ) -> None:
        wf = stencil.turbulence(self.ae, self.seed, parallel=parallel)
        wf.write(
            self.output,
            format=self.format,
            u_offset=self.u_offset,
            y_offset=self.y_offset,
            z_offset=self.z_offset,
        )
