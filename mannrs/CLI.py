import click
from pathlib import Path
from time import perf_counter
from typing import Union

import toml
from pydantic import BaseModel
from rich import print
from .Stencil import Stencil


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
@click.option(
    "--benchmark",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to benchmarking output file",
)
@click.argument("filename", type=click.Path(exists=True, path_type=Path))
def CLI(
    filename: Path,
    parallel: bool,
    dryrun: bool,
    skip_existing: bool,
    benchmark: Union[Path, None],
):
    """
    Mann.rs turbulence generator.
    Author: Jaime Liew <jaimeliew1@gmail.com>
    """

    sim = Stencil.from_toml(filename)

    if dryrun:
        print("[DRY RUN] Input file successfully read. Skipping turbulence generation.")
        print("Parsed simulation parameters:")
        print(sim)
        return

    if skip_existing and all(x.output.exists() for x in sim.turbulence_boxes):
        print("All turbulence boxes already exist. Skipping generation.")
        return

    if sim.constrained:
        print(
            f"Generating constrained stencil with {len(sim.constraint_spec.constraints)} constraints..."
        )
        print(sim.stencil_spec)
        print(sim.constraint_spec)
    else:
        print("Generating unconstrained stencil...")
        print(sim.stencil_spec)

    stencil = sim.build(parallel=parallel)

    if sim.constrained:
        print(f"Correlation matrix sparsity: {100*stencil.sparsity:.4f} %")
        print(f"Spectral compression: {100*stencil.spectral_compression:.4f} %")
    print(f"Stencil generated in {stencil.stencil_time:.4f} seconds.\n")

    turb_times: list[float] = []
    for i, turbbox in enumerate(sim.turbulence_boxes, start=1):
        print(f"Generating turbulence box {i}/{len(sim.turbulence_boxes)}...")
        print(f"Parameters: {turbbox}")
        if skip_existing and turbbox.output.exists():
            print(f"Output '{turbbox.output}' already exists. Skipping.")
            continue
        tstart = perf_counter()
        turbbox.generate_and_save(stencil, parallel=parallel)

        turb_times.append(perf_counter() - tstart)
        print(f"Turbulence box {i} generated in {turb_times[-1]:.4f} seconds.\n")

    if benchmark:
        Benchmark(
            stencil_time=stencil.stencil_time,
            sparsity=stencil.sparsity,
            spectral_compression=stencil.spectral_compression,
            turb_times=turb_times,
        ).to_toml(benchmark)


class Benchmark(BaseModel):
    stencil_time: float
    sparsity: Union[float, None] = None
    spectral_compression: Union[float, None] = None
    turb_times: list[float]

    def to_toml(self, fn: Path) -> None:
        with open(fn, "w") as f:
            toml.dump(self.model_dump(), f)
