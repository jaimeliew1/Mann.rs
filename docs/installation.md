# Installation

Mann.rs is available on **Windows**, **Linux**, and **MacOS**. It can be installed in the form of a Python package from Pypi:

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
## Rust backend installation
The heavy computations in Mann.rs are written in Rust. The underlying rust package can be installed or added as a Rust dependency. Note: the commandline interface is not available in the Rust version.
```bash
cargo install --git https://github.com/jaimeliew1/Mann.rs mannrs
```