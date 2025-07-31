[![DOI](https://zenodo.org/badge/450532624.svg)](https://zenodo.org/badge/latestdoi/450532624)

# Mann.rs
A constrained Mann turbulence generator for Python written in Rust. Mannrs can generate 3D coherent turbulence boxes for wind turbine simulations. It can generate 3D coherent turbulence boxes for wind turbine simulations, including constrained turbulence fields that enforce any number of velocity constraints at arbitrary locations within the wind field. The numerical innovations used in this package are described in [Liew, J., Riva, R., & Göçmen, T. (2023) *Efficient Mann turbulence generation for offshore wind farms with applications in fatigue load surrogate modelling*](https://doi.org/10.1088/1742-6596/2626/1/012050). The underlying Mann turbulence model is originially described in [Mann, J. (1998). *Wind field simulation*](https://doi.org/10.1016/S0266-8920(97)00036-2).

Features include:
- **Parallelized computations:** Just set `parallel=True`
- **Memory efficient:** Can generate extremely high resolution turbulence.
- **Blazing fast:** Thanks to the stencil method and the Rust backend.
- **Arbitrary box sizing:** Box discretization is not limited to powers of 2.
- **Supports constrained turbulence:** Generate turbulence fields conditioned on pointwise velocity constraints.

# Usage
Mannrs separates the turbulence generation process into two distinct steps:

1) **Stencil generation** — computes a reusable turbulence stencil object that can efficiently generate 3D turbulent wind fields.

2) **Turbulence synthesis** — generates velocity fields from the stencil given a random seed.

This design enables reusing the stencil to generate multiple independent turbulence realizations efficiently.

## Python

### Generating unconstrained turbulence
```python
import mannrs

# Define parameters
params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000,
    "Ly": 200,
    "Lz": 200,
    "Nx": 8192,
    "Ny": 64,
    "Nz": 64,
}
ae = 0.2
seed = 1234

# Generate stencil
stencil = mannrs.Stencil(**params)

# Generate one turbulence realization
x, y, z = stencil.get_axes()
U, V, W = stencil.turbulence(ae, seed)
```


### Generating constrained turbulence
```python
from mannrs import Constraint, ConstrainedStencil

# Define velocity constraints at spatial points
constraints = [
    Constraint(x=100.0, y=50.0, z=10.0, u=8.5),
    Constraint(x=200.0, y=60.0, z=15.0, u=7.9),
]

# Define parameters as before
params = {
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000,
    "Ly": 200,
    "Lz": 200,
    "Nx": 8192,
    "Ny": 64,
    "Nz": 64,
}

# Create a constrained stencil
cstencil = ConstrainedStencil(constraints=constraints, **params)

# Generate constrained turbulence realization
x, y, z = cstencil.get_axes()
U, V, W = cstencil.turbulence(ae=0.2, seed=1234)
```


## Rust
```rust
use mannrs::Stencil;

let (L, gamma) = (30.0, 3.2);
let (Lx, Ly, Lz) = (6000.0, 200.0, 200.0);
let (Nx, Ny, Nz) = (8192, 64, 64);
let (aperiodic_x, aperiodic_y, aperiodic_z) = (false, true, true);
let parallel = true;
let ae = 0.2;
let seed = 1234;


let stencil = Stencil::from_params(                
                L,
                gamma,
                Lx,
                Ly,
                Lz,
                Nx,
                Ny,
                Nz,
                aperiodic_x,
                aperiodic_y,
                aperiodic_z,
                sinc_thres,
                parallel,
            );

let (U, V, W) = stencil.turbulence(ae, seed, parallel);
```

# Installation
## Installation from Pypi
Mann.rs supports Windows, Linux, and MacOS, and can be installed easily via pip without requiring a Rust compiler:
```bash
pip install mannrs
```
## Installation from source
To install from source, the Rust compiler must be installed. Clone the repository and run pip install:
```bash
git clone git@github.com:jaimeliew1/Mann.rs.git
cd Mann.rs
pip install .
```

## Installation in Rust
To install the underlying Rust package for use in Rust, run:
```bash
cargo install --git https://github.com/jaimeliew1/Mann.rs mannrs
```

# Contributions
If you have suggestions or issues with Mann.rs, feel free to contact me at `jaimeliew1@gmail.com`. Pull requests are welcome.

# Citation
If you want to cite Mann.rs, please use this citation:
```
Jaime Liew. (2022). jaimeliew1/Mann.rs: Publish Mann.rs (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.7254149
```
