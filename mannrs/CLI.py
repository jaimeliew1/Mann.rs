import click
from pathlib import Path


from .InputFile import MannrsInputParams, run


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
    benchmark: Path | None,
):
    """
    Mann.rs turbulence generator.
    Author: Jaime Liew <jaimeliew1@gmail.com>
    """

    sim = MannrsInputParams.from_toml(filename)
    run(sim, parallel, dryrun, skip_existing, benchmark)
