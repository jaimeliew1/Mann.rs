# RustMann
A Mann turbulence generator for Python written in Rust. RustMann can generate 3D coherent turbulence boxes for wind turbine simulations as described in *Mann, J. (1998). Wind field simulation. Probabilistic engineering mechanics, 13(4), 269-282.*

Features include:
- **Parallelized computations:** Just set `parallel=True`
- **Memory efficient:** Can generate extremely high resolution turbulence.
- **Blazing fast:** Thanks to the stencil method and the Rust backend.
- **Arbitrary box sizing:** Box discretization is not limited to powers of 2.

# Usage
RustMann separates the process of generating turbulence into two steps: **stencil generation** and **turbulence generation**. The stencil is a 5D matrix containing the spectral tensors needed to generate turbulence for a given set of parameters. A stencil can be reused to generate multiple random instances of turbulence. Implementations are provided in Python and Rust.

## Python
```python
import RustMann

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

stencil = RustMann.Stencil(**params)
U, V, W = stencil.turbulence(ae, seed)
```

## Rust
```rust
use RustMann::Stencil;

let (L, gamma) = (30.0, 3.2);
let (Lx, Ly, Lz) = (6000.0, 200.0, 200.0);
let (Nx, Ny, Nz) = (8192, 64, 64);
let ae = 0.2;
let seed = 1234;

let stencil = Stencil::from_params(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz);
let (U, V, W) = stencil.turbulence(ae, seed);
```

# Installation
## Python (Linux and MacOS only)
```bash
pip install RustMann
```

## Rust
Add this to your `Cargo.toml`:
```toml
[dependencies]
rustmann = "0.1.0"
```
