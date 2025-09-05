from __future__ import annotations
from pathlib import Path
from time import perf_counter


import toml
from pydantic import BaseModel, Field
from typing import Optional, Literal
from rich import print

from . import Stencil, ConstrainedStencil, Constraint


class SimConstraint(BaseModel):
    x: float
    y: float
    z: float
    u: float


class TurbulenceParams(BaseModel):
    ae: float
    seed: int
    output: Path
    format: Literal["npz", "netCDF", "HAWC2"] = "npz"
    u_offset: float = 0.0
    """Velocity offset added to the u velocity component."""
    y_offset: float = 0.0
    """Spatial offset added to the y axis (e.g. to center the box at zero.)"""
    z_offset: float = 0.0
    """Spatial offset added to the z axis."""


class StencilParams(BaseModel):
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
    """
    Threshold for applying the Mann sinc correction to low-frequency modes. 
    """


class ConstraintParams(BaseModel):
    constraints: list[SimConstraint] = Field(..., repr=False)
    spectral_compression_target: float = 0.8
    """Desired compression ratio for the constraint impulse response."""
    corr_thres: float = 0.0001
    """Threshold for sparsifying the constraint correlation matrix"""


class MannrsInputParams(BaseModel):
    stencil_params: StencilParams
    turbulence_boxes: list[TurbulenceParams]
    constraint_params: Optional[ConstraintParams] = None

    @classmethod
    def from_toml(cls, fn: Path) -> MannrsInputParams:
        with open(fn, "r") as f:
            data = toml.load(f)
        sim = MannrsInputParams(**data)
        return sim


def generate_stencil(
    sim: MannrsInputParams, parallel: bool
) -> Stencil | ConstrainedStencil:
    """Generate a turbulence stencil based on simulation parameters, optionally with constraints."""
    if sim.constraint_params is None:
        print("Generating unconstrained stencil...")
        print(f"Stencil parameters: {sim.stencil_params}")
        stencil = Stencil(**sim.stencil_params.model_dump(), parallel=parallel)
    else:
        constraints = [
            Constraint(x.x, x.y, x.z, x.u) for x in sim.constraint_params.constraints
        ]
        print(f"Generating constrained stencil with {len(constraints)} constraints...")
        print(f"Stencil parameters: {sim.stencil_params}")
        print(f"Constraint parameters: {sim.constraint_params}")
        stencil = ConstrainedStencil(
            **sim.stencil_params.model_dump(),
            spectral_compression_target=sim.constraint_params.spectral_compression_target,
            corr_thres=sim.constraint_params.corr_thres,
            constraints=constraints,
            parallel=parallel,
        )
        print(f"Correlation matrix sparsity: {stencil.sparsity}")
        print(f"Spectral compression: {stencil.spectral_compression}")

    return stencil


def generate_turbulence_boxes(
    sim: MannrsInputParams, stencil: Stencil, parallel: bool, skip_existing: bool
) -> list[float]:
    """Generate all turbulent wind fields and write output in the specified format."""

    turb_times: list[float] = []
    for i, turbbox in enumerate(sim.turbulence_boxes, start=1):
        print(f"Generating turbulence box {i}/{len(sim.turbulence_boxes)}...")
        print(f"Parameters: {turbbox}")
        if skip_existing and turbbox.output.exists():
            print(f"Output '{turbbox.output}' already exists. Skipping.")
            continue
        tstart = perf_counter()
        turb = stencil.turbulence(turbbox.ae, turbbox.seed, parallel=parallel)


        if turbbox.format== "npz":
            turb.to_npz(
                turbbox.output,
                U_offset=turbbox.u_offset,
                y_offset=turbbox.y_offset,
                z_offset=turbbox.z_offset,
            )
            print(f"Output written to '{turbbox.output}' (npz format).")
        elif turbbox.format== "netCDF":
            turb.to_netCDF(
                turbbox.output,
                Uamb=0.0,
                U_offset=turbbox.u_offset,
                y_offset=turbbox.y_offset,
                z_offset=turbbox.z_offset,
            )
            print(f"Output written to '{turbbox.output}' (netCDF format).")
        elif turbbox.format== "HAWC2":
            _stem = turbbox.output.stem
            turb.to_HAWC2(
                turbbox.output.with_stem(_stem + "_u"),
                turbbox.output.with_stem(_stem + "_v"),
                turbbox.output.with_stem(_stem + "_w"),
                U_offset=turbbox.u_offset,
            )
            print(f"Output written to '{turbbox.output}' (HAWC2 format).")
        else:
            raise ValueError(f"ERROR Output format '{turbbox.format}' not implemented.")

        turb_times.append(perf_counter() - tstart)
        print(f"Turbulence box {i} generated in {turb_times[-1]:.4f} seconds.\n")

    return turb_times


class Benchmark(BaseModel):
    stencil_time: float
    sparsity: float | None = None
    spectral_compression: float | None = None
    turb_times: list[float]

    def to_toml(self, fn: Path) -> None:
        with open(fn, "w") as f:
            toml.dump(self.model_dump(), f)


def run(
    sim: MannrsInputParams,
    parallel: bool,
    dryrun: bool,
    skip_existing: bool,
    benchmark: Path | None,
):
    if dryrun:
        print("[DRY RUN] Input file successfully read. Skipping turbulence generation.")
        print("Parsed simulation parameters:")
        print(sim)
        return

    if skip_existing and all(x.output.exists() for x in sim.turbulence_boxes):
        print("All turbulence boxes already exist. Skipping generation.")
        return

    tstart = perf_counter()
    stencil = generate_stencil(sim, parallel)
    stencil_time = perf_counter() - tstart
    print(f"Stencil generated in {stencil_time:.4f} seconds.\n")

    turb_times = generate_turbulence_boxes(sim, stencil, parallel, skip_existing)

    if benchmark:
        if isinstance(stencil, ConstrainedStencil):
            sparsity = stencil.sparsity
            spectral_compression = stencil.spectral_compression
        else:
            sparsity, spectral_compression = None, None
        Benchmark(
            stencil_time=stencil_time,
            sparsity=sparsity,
            spectral_compression=spectral_compression,
            turb_times=turb_times,
        ).to_toml(benchmark)
