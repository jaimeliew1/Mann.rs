# Understanding the Stencil: Efficient Batch Generation of Mann Turbulence

## What is a Stencil?

In the context of Mann turbulence generation, a **stencil** represents the pre-computed spectral structure that defines the statistical properties of the turbulence field. Think of it as a reusable template or blueprint that encodes

The stencil contains all the computationally expensive setup work needed to generate turbulence with specific geometric and physical parameters. Once built, it can be reused to generate multiple turbulent realizations with different turbulence intensities and random seeds efficiently.


## Basic Batch Generation

Here's how to efficiently generate multiple unconstrained turbulent boxes with the same statistical properties:

```python
from mannrs import Stencil

# Define your turbulence parameters once
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

# Build the stencil once
stencil = Stencil(**mann_params).build()

# Generate multiple realizations efficiently
ae = 0.2

for seed in range(10):  # Generate 10 different turbulent boxes using different seeds
    filename = f"turbulence_box_{seed:03d}.npz"
    
    # Fast generation using the pre-built stencil
    stencil.turbulence(ae, seed).write(filename)
    print(f"Generated {filename}")
```

## Constrained Batch Generation

For constrained turbulence, a single stencil built with specific constraints can be reused to produce various turbulent realizations.

```python
from mannrs import Stencil, Constraint

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

# Define constraints
constraints = [
    Constraint(x=1000.0, y=0.0, z=100.0, u=8.5),
    Constraint(x=2000.0, y=0.0, z=100.0, u=8.3),
    Constraint(x=3000.0, y=0.0, z=100.0, u=8.1),
]

# Build constrained stencil once (expensive operation)
stencil = Stencil(**mann_params).constrain(constraints).build()

# Generate multiple constrained realizations efficiently
ae = 0.2

for seed in range(5):
    filename = f"constrained_turbulence_{seed:03d}.npz"
    stencil.turbulence(ae, seed).write(filename)
    print(f"Generated constrained box: {filename}")
```
# Technical Description of the Stencil
## Unconstrained Stencil
The unconstrained stencil contains a 5-dimensional array that stores the spectral tensor decompositions for all wavenumbers in 3D space. This array encodes the complete statistical structure of the Mann turbulence model, including the energy distribution across different scales and the correlations between velocity components. Once computed, this spectral information can be combined with random phases to generate multiple independent turbulent realizations.

## Constrained Stencil
The constrained stencil includes additional precomputed data structures to efficiently handle velocity constraints:

1. **LU-Factorized Correlation Matrix**

This contains the decomposed correlation matrix between all constraint positions. The LU factorization enables fast solution of the linear system that determines constraint weights for each turbulent realization, avoiding expensive matrix inversions during generation.

2. **Compressed Spectral Impulse Response Fields**

For each constraint point, the stencil stores compressed frequency-domain impulse response fields. During generation, these fields are:

- Phase-shifted to match each constraint's spatial position

- Scaled by the computed constraint weights

- Summed together using parallel processing to create the constraint contribution

The spectral compression reduces storage requirements while maintaining accuracy. High compression ratios (typically ~95%) can be achieved with negligible impact on the final results, making constrained generation both memory-efficient and fast.