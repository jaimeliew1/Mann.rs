
# Basic Usage
## Turbulence from command line
The **Mann.rs** package can be run directly from the command line by providing an input file in TOML format:
```bash
mannrs input.toml
```


Example input file:
```toml
[stencil_spec]
L = 30.0    
gamma = 3.9 
Lx = 1000.0
Ly = 1000.0
Lz = 1000.0
Nx = 128   
Ny = 64    
Nz = 64    

[[turbulence_boxes]]
ae = 0.05                
seed = 42 

[[turbulence_boxes]]
ae = 0.05                
seed = 123 
```

This input file creates two turbulent wind fields with the same Mann parameters, but with two different random seeds.

### Verify an input file
Use `--dryrun` to check that an input file is valid without generating turbulence:
```bash
mannrs --dryrun input.toml
```

### Turn off parallelisation
By default, Mann.rs uses parallelisation. To force serial execution:
```bash
mannrs --serial input.toml
```

### Avoid Overwriting Existing Files
**Mann.rs** will overwrite any existing turbulent wind field files. Use the `--skip-existing` option to leave existing results untouched and only generate missing outputs:
```bash
mannrs --skip-existing input.toml
```
This is especially useful when running batch jobs, where some files may have already been created.

### See all available options
To list all available command-line options and their descriptions, use the `--help` flag:
```bash
mannrs --help
```

## Turbulence from Python

### Building a stencil
A Stencil defines the turbulence generation setup. There are three ways to build one.

**1. Unconstrained Stencil**
   
An unconstrained stencil can be built directly from Mann parameters:
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

stencil = Stencil(**mann_params).build()
```
**2. Constrained Stencil**
   
Constraints can be added to force the turbulence field to match given velocity values at specific locations:
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
constraints = [
    Constraint(x=1000.0, y=0.0, z=100.0, u=8.5),
    Constraint(x=2000.0, y=0.0, z=100.0, u=8.3),
    Constraint(x=3000.0, y=0.0, z=100.0, u=8.1),
]
stencil = Stencil(**mann_params).constrain(constraints).build()
```

Each constraint specifies a position and the desired streamwise velocity:

```python
constraint = Constraint(
    x=1000.0,    # x-position (m)
    y=0.0,       # y-position (m)
    z=100.0,     # z-position (m)
    u=8.5,       # u-velocity component (m/s)
)

```
**3. From a TOML File**

A stencil can also be built from an input TOML file (either constrained or unconstrained):
```python
from mannrs import Stencil

stencil = Stencil.from_toml("input.toml").build()
```

### Generating turbulence
Once a stencil is built, turbulence fields can be generated with the `turbulence()` method. The two main arguments are:
- `ae`: turbulence scaling parameter
- `seed`: random number generator seed

```python
ae, seed = 0.1, 12345
windfield = stencil.turbulence(ae, seed)
```
Here is a basic example of generating unconstrained turbulence with Mann.rs:

### Saving turbulent wind fields

the `turbulence()` method returns a **Windfield** object, which contains the three components of velocity as well as the axes as Numpy arrays:

```python
windfield.U  # U velocity component (Nx x Ny x Nz)
windfield.V  # V velocity component (Nx x Ny x Nz)
windfield.W  # W velocity component (Nx x Ny x Nz)

windfield.x  # x axis (Nx)
windfield.y  # y axis (Ny)
windfield.z  # z axis (Nz)

```

The wind field can be written to file. By default, they are written in Numpy archive format (`npz`):
```python
windfield.write("output.npz")
```
Other supported formats include **HAWC2** and **netCDF**:
```python
windfield.write("output.nc", format="netCDF")
```
💡 Missing a format you need? Let me know.

### Adding a Streamwise Offset

Mann.rs wind fields have zero-mean velocities. To set a non-zero mean wind speed, provide a `U_offset`:

```python
windfield.write("output.npz", U_offset=10.0) # mean wind = 10 m/s
```
### Shifting Axes
All Mann.rs wind field axes start at zero:

- `x`: $0 \rightarrow L_x$
  
- `y`: $0 \rightarrow L_y$
  
- `z`: $0 \rightarrow L_z$
  
Offsets can be applied during saving:
```python
windfield.write("output.npz", y_offset=-100.0, z_offset=20.0)
```

## Complete example (Constraned Turbulence)



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
    Stencil(**mann_params)    # Create a stencil with Mann parameters
    .constrain(constraints)   # Apply a list of constraints
    .build()                  # Build the turbulence stencil instance
    .turbulence(ae, seed)     # Generate turbulent wind field
    .write(filename)          # Save windfield to file
)
```