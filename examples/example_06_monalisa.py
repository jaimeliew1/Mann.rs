from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from rich import print

from mannrs import ConstrainedStencil, Constraint

FIGDIR = Path("fig")
FIGDIR.mkdir(parents=True, exist_ok=True)

IMAGE_FN = Path(__file__).parent / "mona_lisa.webp"
# METHOD = "fastinterp"
# METHOD = "python_reduced"
METHOD = "rust"
PARALLEL = True
ae, L, gamma = 0.05, 30, 3.9
Lx, Ly, Lz = 1000, 1000, 1000
Nx, Ny, Nz = 128, 64, 64
RES = 60
sinc_thres=3.0
if __name__ == "__main__":
    # Load mona lisa image as an array zero-mean array.
    img = Image.open(IMAGE_FN).resize((RES, RES)).convert("L")
    arr = np.array(img)
    arr = (arr - arr.mean()) / np.std(arr) * 5
    print(arr.min(), arr.max(), arr.mean())

    # Convert mona lisa pixels to wind field constraints.
    constraints = []
    for i, y in enumerate(np.linspace(0, Ly, RES)):
        for j, z in enumerate(np.linspace(0, Lz, RES)):
            constraints.append(Constraint(Lx / 2, y, z, arr[i, j], None, None))

    print(len(constraints))
    stencil = ConstrainedStencil(
        constraints,
        ae,
        L,
        gamma,
        Nx,
        Ny,
        Nz,
        Lx,
        Ly,
        Lz,
        parallel=PARALLEL,
        aperiodic_x=False,
        sinc_thres=sinc_thres,
    )
    print(stencil.stencil)
    U, V, W = stencil.turbulence(1234, method=METHOD, parallel=PARALLEL, thres=0.0005)

    for i, slice in enumerate(tqdm(U)):
        if i % 16 != 0:
            continue
        plt.figure()
        plt.imshow(slice)
        plt.savefig(FIGDIR / f"{i:03d}.png", dpi=300)
        plt.close()
