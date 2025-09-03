from pathlib import Path
from time import perf_counter

import click
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
    format: Literal["netCDF", "HAWC2"] = "netCDF"


class StencilParams(BaseModel):
    L: float
    gamma: float
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int
    aperiodic_x: bool = False
    aperiodic_y: bool = True
    aperiodic_z: bool = True
    sinc_thres: float = 3.0


class ConstraintParams(BaseModel):
    constraints: list[SimConstraint] = Field(..., repr=False)
    spectral_compression_target: float = 0.8
    corr_thres: float = 0.0001


class SimulationParams(BaseModel):
    stencil_params: StencilParams
    turbulence_boxes: list[TurbulenceParams]
    constraint_params: Optional[ConstraintParams] = None


@click.command()
@click.option(
    "--parallel/--serial",
    default=True,
    help="Parallelize stencil and turbulence generation.",
    show_default=True,
)
@click.option(
    "--dryrun",
    is_flag=True,
    default=False,
    help="Evaluate input files without generating turbulence.",
)
@click.argument("filename", type=click.Path(exists=True, path_type=Path))
def CLI(filename, parallel, dryrun):
    """
    Mann.rs turbulence generator.
    Author: Jaime Liew <jaimeliew1@gmail.com>
    """

    main(filename, parallel, dryrun)


def main(src, parallel, dryrun):
    # Load input file and parse into SimulationParams

    with open(src, "r") as f:
        data = toml.load(f)
    sim = SimulationParams(**data)

    if dryrun:
        print("[DRY RUN] Input file successfully read. Skipping turbulence generation.")
        print("Parsed simulation parameters:")
        print(sim)
        return

    # Generate stencil (constrained or unconstrained)
    tstart = perf_counter()
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
    stencil_time = perf_counter() - tstart
    print(f"Stencil generated in {stencil_time:.4f} seconds.\n")

    # Generate turbulence for each box
    for i, turbbox in enumerate(sim.turbulence_boxes, start=1):
        print(f"Generating turbulence box {i}/{len(sim.turbulence_boxes)}...")
        print(f"Parameters: {turbbox}")
        tstart = perf_counter()
        turb = stencil.turbulence(turbbox.ae, turbbox.seed, parallel=parallel)

        match turbbox.format:
            case "netCDF":
                turb.to_netCDF(turbbox.output, Uamb=0.0)
                print(f"Output written to '{turbbox.output}' (netCDF format).")
            case "HAWC2":
                _stem = turbbox.output.stem
                turb.to_HAWC2(
                    turbbox.output.with_stem(_stem + "_u"),
                    turbbox.output.with_stem(_stem + "_v"),
                    turbbox.output.with_stem(_stem + "_w"),
                )
                print(f"Output written to '{turbbox.output}' (HAWC2 format).")
            case other:
                raise ValueError(f"ERROR Output format '{other}' not implemented.")

        turb_time = perf_counter() - tstart
        print(f"Turbulence box {i} generated in {turb_time:.4f} seconds.\n")
