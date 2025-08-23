"""
Image-Constrained Turbulence Field Generator

Converts images into constraint fields for turbulence generation.
The main function `encode_image_as_constraint_field()` places image-based
constraints at specified x-locations within a 3D computational domain.

In this example, the Mona Lisa is embedded within a turbulent wind field.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from mannrs import ConstrainedStencil, Constraint


IMAGE_FN = Path(__file__).parent / "mona_lisa.webp"

ae, L, gamma = 0.05, 30, 3.9
Lx, Ly, Lz = 1000, 1000, 1000
Nx, Ny, Nz = 128, 64, 64
RES = 60


def encode_image_as_constraint_field(
    image_path: Path,
    x_location: float,
    Ly: float,
    Lz: float,
    resolution: int = 60,
    intensity_scale: float = 5,
) -> list[Constraint]:
    """
    Encode an image as a constraint field at a particular x location.

    Parameters:
    -----------
    image_path : Path or str
        Path to the image file
    x_location : float
        X coordinate where the constraint field should be placed
    Ly : float
        Y dimension of the domain
    Lz : float
        Z dimension of the domain
    resolution : int, default=60
        Resolution to resize the image to (creates resolution x resolution grid)
    intensity_scale : float, default=5
        Scaling factor for the constraint intensities

    Returns:
    --------
    List[Constraint]
        List of Constraint objects representing the image
    """
    # Load and preprocess image
    img = Image.open(image_path).resize((resolution, resolution)).convert("L")
    arr = np.array(img)

    # Normalize to zero-mean and scale
    arr = (arr - arr.mean()) / np.std(arr) * intensity_scale

    # Convert image pixels to constraints
    constraints: list[Constraint] = []
    y_coords: np.ndarray = np.linspace(0, Ly, resolution)
    z_coords: np.ndarray = np.linspace(0, Lz, resolution)

    for i, y in enumerate(y_coords):
        for j, z in enumerate(z_coords):
            constraints.append(Constraint(x_location, y, z, arr[i, j]))

    return constraints


if __name__ == "__main__":
    # Load mona lisa image as an array zero-mean array.
    constraints = encode_image_as_constraint_field(IMAGE_FN, Lx / 2, Ly, Lz, RES)

    print(f"Generating turbulence stencil with {len(constraints)} constraints...")
    stencil = ConstrainedStencil(
        constraints, L, gamma, Nx, Ny, Nz, Lx, Ly, Lz, spectral_compression_target=0.9
    )

    print(f"Correlation matrix sparsity: {100 * stencil.sparsity:.2f}%")
    print(f"Spectral compression: {100 * stencil.spectral_compression:.2f}%")

    print("Generating constrained turbulence...")
    U, V, W = stencil.turbulence(ae, 1234, parallel=True)

    print("Plotting...")
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for slice, ax in zip(U[::16], axes):
        ax.imshow(slice)
        ax.axis("off")

    print("Done!")
    plt.show()
