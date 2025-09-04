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
    format: Literal["npz", "netCDF", "HAWC2"] = "npz"


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
    show_default=True,
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Do not overwrite existing files.",
    show_default=True,
)
@click.argument("filename", type=click.Path(exists=True, path_type=Path))
def CLI(filename: Path, parallel: bool, dryrun: bool, skip_existing: bool):
    """
    Mann.rs turbulence generator.
    Author: Jaime Liew <jaimeliew1@gmail.com>
    """

    main(filename, parallel, dryrun, skip_existing)


def load_sim_params(src: Path) -> SimulationParams:
    """Load and parse the TOML input file into a SimulationParams object."""
    with open(src, "r") as f:
        data = toml.load(f)
    sim = SimulationParams(**data)
    return sim


def generate_stencil(
    sim: SimulationParams, parallel: bool
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
    sim: SimulationParams, stencil: Stencil, parallel: bool, skip_existing: bool
) -> None:
    """Generate all turbulent wind fields and write output in the specified format."""
    for i, turbbox in enumerate(sim.turbulence_boxes, start=1):
        print(f"Generating turbulence box {i}/{len(sim.turbulence_boxes)}...")
        print(f"Parameters: {turbbox}")
        if skip_existing and turbbox.output.exists():
            print(f"Output '{turbbox.output}' already exists. Skipping.")
            continue
        tstart = perf_counter()
        turb = stencil.turbulence(turbbox.ae, turbbox.seed, parallel=parallel)

        match turbbox.format:
            case "npz":
                turb.to_npz(turbbox.output)
                print(f"Output written to '{turbbox.output}' (npz format).")
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


def main(src: Path, parallel: bool, dryrun: bool, skip_existing: bool):
    sim = load_sim_params(src)

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

    generate_turbulence_boxes(sim, stencil, parallel, skip_existing)
