[![DOI](https://zenodo.org/badge/450532624.svg)](https://zenodo.org/badge/latestdoi/450532624)

# Mann.rs
A Mann turbulence generator for Python written in Rust. Mannrs can generate 3D coherent turbulence boxes for wind turbine simulations as described in *Mann, J. (1998). Wind field simulation. Probabilistic engineering mechanics, 13(4), 269-282.*

Features include:
- **Parallelized computations:** Just set `parallel=True`
- **Memory efficient:** Can generate extremely high resolution turbulence.
- **Blazing fast:** Thanks to the stencil method and the Rust backend.
- **Arbitrary box sizing:** Box discretization is not limited to powers of 2.

# Usage
Mannrs separates the process of generating turbulence into two steps: **stencil generation** and **turbulence generation**. The stencil is a 5D matrix containing the spectral tensors needed to generate turbulence for a given set of parameters. A stencil can be reused to generate multiple random instances of turbulence. Implementations are provided in Python and Rust.

## Python
```python
import mannrs

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

stencil = mannrs.Stencil(**params)
U, V, W = stencil.turbulence(ae, seed)
```

## Rust
```rust
use mannrs::Stencil;

let (L, gamma) = (30.0, 3.2);
let (Lx, Ly, Lz) = (6000.0, 200.0, 200.0);
let (Nx, Ny, Nz) = (8192, 64, 64);
let ae = 0.2;
let seed = 1234;

let stencil = Stencil::from_params(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz);
let (U, V, W) = stencil.turbulence(ae, seed);
```

# Installation
Installation for both Python and Rust versions requires the Rust compiler to be installed (see [here](https://www.rust-lang.org/tools/install) for installation instructions).
## Python (Linux and MacOS only)


Clone this repository and pip install:
```bash
git clone git@github.com:jaimeliew1/Mann.rs.git
cd Mann.rs
pip install .
```

## Rust
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