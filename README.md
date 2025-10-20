# Mann.rs

[![DOI](https://zenodo.org/badge/450532624.svg)](https://zenodo.org/badge/latestdoi/450532624)
[![PyPI version](https://badge.fury.io/py/mannrs.svg)](https://badge.fury.io/py/mannrs)
[![PyPI downloads](https://img.shields.io/pypi/dm/mannrs.svg)](https://pypi.org/project/mannrs/)
[![GitHub stars](https://img.shields.io/github/stars/jaimeliew1/mann.rs.svg?style=social)](https://github.com/jaimeliew1/Mann.rs)
[![GitHub forks](https://img.shields.io/github/forks/jaimeliew1/mann.rs.svg?style=social)](https://github.com/jaimeliew1/Mann.rs)


**Mann.rs** is a high-performance turbulent wind field generator based on the Mann turbulence model, designed for wind turbine and wind farm simulations. It produces three-dimensional coherent wind fields and supports both unconstrained and constrained turbulence generation.

Built in Rust for speed and efficiency, Mann.rs provides seamless Python bindings and a command-line interface for easy integration and scalability into engineering workflows.

For more information, see the [**Documentation**](https://jaimeliew1.github.io/Mann.rs/).

## **Installation**
**Mann.rs** is available for **Windows**, **MacOS**, and **Linux** as a Python package.
```bash
pip install mannrs
```
For more details on the installation process, see the [**installation Guide**](https://jaimeliew1.github.io/Mann.rs/installation/).
## **Usage**

### **Command line**
```bash
mannrs input.toml
```
Define your simulation parameters in a TOML file. See the [**Input file format**](https://jaimeliew1.github.io/Mann.rs/api/schema/) for details.

### **Python**

```python
from mannrs import Stencil

...

(
    Stencil(**mann_params)    # Define a stencil with Mann parameters
    .constrain(constraints)   # Apply velocity constraints (optional)
    .build()                  # Build the turbulence stencil
    .turbulence(ae, seed)     # Generate turbulent wind field
    .write("out.npz")         # Save windfield to file
)

```
For a step-by-step walkthrough, visit the [**Basic usage**](https://jaimeliew1.github.io/Mann.rs/usage/basic_usage/) page.



# Contributions
If you have suggestions or issues with Mann.rs, feel free raise an issue in the [Mann.rs Github repository](https://github.com/jaimeliew1/Mann.rs). Pull requests are welcome.

# Citation
If you want to cite Mann.rs, please use this citation:
```
Liew, J., Riva, R., & Göçmen, T. (2023). Efficient Mann turbulence generation for offshore wind farms with applications in fatigue load surrogate modelling. Journal of Physics: Conference Series, 2626, 012050. DOI: 10.1088/1742-6596/2626/1/012050
```
