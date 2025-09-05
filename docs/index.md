# Mann.rs

[![DOI](https://zenodo.org/badge/450532624.svg)](https://zenodo.org/badge/latestdoi/450532624)
[![PyPI version](https://badge.fury.io/py/mannrs.svg)](https://badge.fury.io/py/mannrs)


Mann.rs is a Rust-based turbulence generator with Python bindings. It implements the Mann turbulence model to produce three-dimensional coherent wind fields for wind turbine simulations. The library supports both [unconstrained](usage/unconstrained.md) and [constrained](usage/constrained.md) turbulence generation.


## Key Features

- **🚀 Blazing Fast**: Thanks to the stencil method and Rust backend
- **⚡ Parallelized**: Calculations parallelized using Rayon
- **💾 Memory Efficient**: Can generate extremely high resolution turbulence
- **📐 Flexible**: Arbitrary box sizing - not limited to powers of 2
- **🎯 Constrained**: Generate turbulence fields with pointwise velocity constraints




## Scientific Background
The numerical innovations in Mann.rs are described in:

    Liew, J., Riva, R., & Göçmen, T. (2023). Efficient Mann turbulence generation for offshore wind farms with applications in fatigue load surrogate modelling. Journal of Physics: Conference Series, 2626, 012050. DOI: 10.1088/1742-6596/2626/1/012050
The underlying Mann turbulence model is originally described in:

    Mann, J. (1998). Wind field simulation. Probabilistic Engineering Mechanics, 13(4), 269-282. DOI: 10.1016/S0266-8920(97)00036-2