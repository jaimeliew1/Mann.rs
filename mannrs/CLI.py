from pathlib import Path

import click
import toml
from tqdm import tqdm

from . import Stencil, ForgetfulStencil, save_box


@click.command()
@click.option(
    "--parallel/--serial",
    default=True,
    help="Parallelize stencil and turbulence generation.",
    show_default=True,
)
@click.option(
    "--forgetful",
    is_flag=True,
    default=False,
    help="Use low-memory stencil (slower).  [default: off]",
    show_default=True,
)
@click.option(
    "--dryrun",
    is_flag=True,
    default=False,
    help="Evaluate input files without generating turbulence.",
)
@click.argument("src", type=click.Path(exists=True, path_type=Path), nargs=-1)
def CLI(src, forgetful, parallel, dryrun):
    """
    Mann.rs turbulence generator.
    Author: Jaime Liew <jaimeliew1@gmail.com>
    """

    main(src, forgetful, parallel, dryrun)


required_params = [
    "L",
    "ae",
    "gamma",
    "seed",
    "Nx",
    "Ny",
    "Nz",
    "Lx",
    "Ly",
    "Lz",
    "fn_u",
    "fn_v",
    "fn_w",
]
turb_param_keys = ["fn_u", "fn_v", "fn_w", "ae", "seed"]

# https://stackoverflow.com/a/1151705
class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def extract_toml_mann_params(fn):
    """
    Search a toml file (e.g. a HAWC2Farm input file) for Mann box inputs. Return
    list of Mann box parameters found in file.
    """
    data = toml.load(fn)
    out = toml_traverse(data)
    return out


def toml_traverse(input):
    out = []
    if type(input) == dict:
        if all(x in input for x in required_params):
            return [{k: input[k] for k in required_params}]

        for val in input.values():
            if type(val) in [dict, list]:
                out.extend(toml_traverse(val))
    elif type(input) == list:
        for val in input:
            if type(val) in [dict, list]:
                out.extend(toml_traverse(val))

    return out


parsers = {
    ".toml": extract_toml_mann_params,
}


def separate_stencil_and_turb_params(param_list):
    out = {}

    for params in param_list:
        stencil_params = hashabledict(params)
        turb_params = {key: stencil_params.pop(key) for key in turb_param_keys}

        if stencil_params not in out:
            out[stencil_params] = [turb_params]
        else:
            out[stencil_params].append(turb_params)

    return out


def generate_single_separated(
    stencil_params, turb_param_list, parallel=True, forgetful=False, progress_bar=None
):
    """
    Generate turbulence with fixed stencil parameters and a list of turbulence parameters.
    args:
        stencil_params (dict): Mann turbulence stencil parameters.
        turb_param_list (list[dict]): List of turbulence box parameters.
        parallel (bool): Activate parallel execution (default=True)
        forgetful (bool): Activate low-memory stencil (default=False)
        progress_bar (tqdm.tqdm): Progress bar object (default=None)
    return:
        None
    """
    if forgetful:
        stencil = ForgetfulStencil(**stencil_params, parallel=parallel)
    else:
        stencil = Stencil(**stencil_params, parallel=parallel)

    for turb_params in turb_param_list:
        U, V, W = stencil.turbulence(
            turb_params["ae"], turb_params["seed"], parallel=parallel
        )

        save_box(turb_params["fn_u"], U)
        save_box(turb_params["fn_v"], V)
        save_box(turb_params["fn_w"], W)

        if progress_bar:
            progress_bar.update(1)


def main(src, forgetful, parallel, dryrun):

    param_list = []
    for fn in src:

        if fn.suffix in parsers:
            param_list.extend(parsers[fn.suffix](fn))

    # https://stackoverflow.com/questions/11092511/python-list-of-unique-dictionaries
    unique_params = [dict(s) for s in set(frozenset(d.items()) for d in param_list)]

    separated_params = separate_stencil_and_turb_params(unique_params)

    print(
        f"{len(param_list)} turbulence box inputs ({len(unique_params)} unique boxes, {len(separated_params)} unique stencils) found from {len(src)} input files."
    )

    if dryrun:
        return

    progress_bar = tqdm(total=len(unique_params))
    for stencil_params, turb_param_list in separated_params.items():
        generate_single_separated(
            stencil_params,
            turb_param_list,
            parallel=parallel,
            forgetful=forgetful,
            progress_bar=progress_bar,
        )
