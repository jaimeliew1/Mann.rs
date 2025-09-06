# Unconstrained Turbulence

Here is a basic example of generating unconstrained turbulence with Mann.rs:

```python
from mannrs import Stencil

mann_params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000.0,
    "Ly": 200.0,
    "Lz": 200.0,
    "Nx": 8192,
    "Ny": 32,
    "Nz": 32,
}

ae, seed = 0.2, 1234
filename = "output.npz"

Stencil(**mann_params).build().turbulence(ae, seed).write(filename)
```

Let's break this down step by step.

## Import the Stencil Class

First, we import the main `Stencil` class from the `mannrs` package:

```python
from mannrs import Stencil
```

The `Stencil` class is the primary interface for generating Mann turbulence. It handles the configuration and generation of the turbulence field.

## Define Parameters

Next, we define the Mann turbulence parameters in a dictionary:

```python
mann_params = {
    "L": 30.0,      # Turbulence length scale (m)
    "gamma": 3.2,   # Mann model parameter
    "Lx": 6000.0,   # Domain size in x-direction (m)
    "Ly": 200.0,    # Domain size in y-direction (m)
    "Lz": 200.0,    # Domain size in z-direction (m)
    "Nx": 8000,     # Number of grid points in x-direction
    "Ny": 32,       # Number of grid points in y-direction
    "Nz": 32,       # Number of grid points in z-direction
}

ae, seed = 0.2, 1234

```

These parameters control:

- **L**: The characteristic length scale of the turbulence

- **gamma**: A parameter in the Mann model that affects the anisotropy

- **Lx, Ly, Lz**: The physical dimensions of the turbulence box

- **Nx, Ny, Nz**: The grid resolution in each direction

!!! note "Grid resolution values"
    - Grid resolutions are **not restricted to powers of 2** like in many other turbulence generators.
    - For best performance, choose resolutions with **low prime divisors** (2, 3, 5, 7, etc).

- **ae**: Alpha epsilon factor (αε^(2/3)) from Mann turbulence theory, where α is the Kolmogorov constant and ε is the turbulent dissipation rate. 
- **seed**: Seed number of the random number generator to ensure reproducible results.

## Generate and Save Turbulence

Finally, we chain the methods to create, generate, and save the turbulence field:

```python
Stencil(**mann_params).build().turbulence(ae, seed).write(filename)
```

This line:

1. **`Stencil(**mann_params)`**: Creates a new Stencil instance with the specified parameters

2. **`.build()`**: Builds the internal structures needed for turbulence generation

3. **`.turbulence(ae, seed)`**: Generates the turbulence field with the given alpha epsilon factor and seed

4. **`.write(filename)`**: Saves the turbulence field to a NumPy archive file

## Output

The resulting file `output.npz` will contain the three-dimensional velocity field arrays that can be loaded and used in your simulations or analysis.

## Constrained Turbulence

For constrained turbulence generation, you need to define constraint points and include them in the generation process:

```python
from mannrs import Stencil, Constraint

# Same parameters as before
mann_params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000.0,
    "Ly": 200.0,
    "Lz": 200.0,
    "Nx": 8000,
    "Ny": 32,
    "Nz": 32,
}

# Define constraint points
constraints = [
    Constraint(x=1000.0, y=0.0, z=100.0, u=8.5),
    Constraint(x=2000.0, y=0.0, z=100.0, u=8.3),
    Constraint(x=3000.0, y=0.0, z=100.0, u=8.1),
]

ae, seed = 0.2, 1234
filename = "constrained_output.npz"

(
    Stencil(**mann_params)
    .constrain(constraints)
    .build()
    .turbulence(ae, seed)
    .write(filename)
)
```

### Constraint Definition

Each constraint point is defined by the position and the streamwise velocity of the constraint in 3D space:

```python
constraint = Constraint(
    x=1000.0,    # x-position (m)
    y=0.0,       # y-position (m)
    z=100.0,     # z-position (m)
    u=8.5,       # u-velocity component (m/s)
)

```

The constraint points specify exact velocity values that the turbulence field must match at those locations.

### Constrained Generation Process

The constrained turbulence generation follows this workflow:

```python
(
    Stencil(**mann_params)
    .constrain(constraints)
    .build()
    .turbulence(ae, seed)
    .write(filename)
)
```

This differs from unconstrained generation by adding the `.constrain(constraints)` step:

1. **`Stencil(**mann_params)`**: Creates a new Stencil instance

2. **`.constrain(constraints)`**: Applies the constraint points to the generation process

3. **`.build()`**: Builds the internal structures with constraint handling

4. **`.turbulence(ae, seed)`**: Generates constrained turbulence field

5. **`.write(filename)`**: Saves the result

