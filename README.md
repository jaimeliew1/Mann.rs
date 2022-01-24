# RustMann
A Mann turbulence generator for Python written in Rust. RustMann can generate 3D coherent turbulence boxes for wind turbine simulations as described in *Mann, J. (1998). Wind field simulation. Probabilistic engineering mechanics, 13(4), 269-282.*

# Usage
RustMann separates the process of generating turbulence into two steps: stencil generation and turbulence generation. The stencil is a 5D matrix containing the spectral tensors needed to generate turbulence for a given set of parameters. A stencil can be reused to generate multiple random instances of turbulence. Implementations are provided in Python and Rust.

## Python
```python
import RustMann

params = {
    "ae": 0.2,
    "L": 30.0,
    "gamma": 3.2,
    "Lx": 6000,
    "Ly": 200,
    "Lz": 200,
    "Nx": 8192,
    "Ny": 64,
    "Nz": 64,
}
seed = 1234

stencil = RustMann.Stencil(**params)
U, V, W = stencil.turbulence(seed)
```

## Rust
```rust
use RustMann::Stencil;

let (ae, L, gamma) = (0.2, 30.0, 3.2);
let (Lx, Ly, Lz) = (6000.0, 200.0, 200.0);
let (Nx, Ny, Nz) = (8192, 64, 64);
let seed = 1234;

let stencil = Stencil::from_params(ae, L, gamma, Lx, Ly, Lz, Nx, Ny, Nz);
let (U, V, W) = stencil.turbulence(seed);
```

# Installation
## Python
```bash
pip install rustmann
```

## Rust
Add this to your `Cargo.toml`:
```toml
[dependencies]
rustmann = "0.1.0"
```
